import importlib.util
import io
import json
import pathlib
import sys
import tarfile


SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


repack = _load("repack_vessl_dataset")
materialize = _load("materialize_prism_archives")
launcher = _load("wait_and_launch_vessl_training")


class LocalStore:
    def __init__(self, source: pathlib.Path, destination: pathlib.Path):
        self.source = source
        self.destination = destination

    def iter_objects(self, component):
        if component == "metadata":
            paths = [path for path in self.source.iterdir() if path.is_file()]
        else:
            root = self.source / component
            paths = list(root.rglob("*")) if root.exists() else []
        for path in sorted(path for path in paths if path.is_file()):
            relative = path.relative_to(self.source).as_posix()
            yield repack.ObjectInfo(
                key=relative,
                relative_path=relative,
                size=path.stat().st_size,
                mtime=0,
            )

    def download(self, obj, destination):
        destination.write_bytes((self.source / obj.relative_path).read_bytes())

    def upload(self, source, key):
        destination = self.destination / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    def read_json(self, key):
        path = self.destination / key
        return json.loads(path.read_text()) if path.exists() else None


def _dataset(root: pathlib.Path):
    (root / "dataset_manifest.json").write_text('{"version": 1}\n')
    for split in ("train", "validation", "test"):
        (root / split).mkdir()
        for index in range(3):
            (root / split / f"sample-{index}.bin").write_bytes(
                f"{split}-{index}".encode()
            )


def test_repack_and_materialize_round_trip(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "archive"
    work = tmp_path / "work"
    output = tmp_path / "output"
    source.mkdir()
    destination.mkdir()
    _dataset(source)
    store = LocalStore(source, destination)
    state = repack.default_state(11, 22)

    manifest = repack.repack(
        store,
        state,
        work,
        target_bytes=20,
        workers=2,
        retries=2,
    )

    assert manifest["complete"] is True
    assert manifest["totals"]["file_count"] == 10
    materialize.materialize(destination, output, workers=2, verify_sha256=True)
    for path in source.rglob("*"):
        if path.is_file():
            assert (output / path.relative_to(source)).read_bytes() == path.read_bytes()


def test_completed_state_is_resumable_without_duplicate_shards(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "archive"
    source.mkdir()
    destination.mkdir()
    _dataset(source)
    store = LocalStore(source, destination)
    state = repack.default_state(11, 22)
    repack.repack(store, state, tmp_path / "work1", 20, 2, 2)
    tar_names = sorted(path.name for path in destination.glob("*.tar"))

    state["complete"] = False
    state["components"]["test"]["complete"] = False
    repack.repack(store, state, tmp_path / "work2", 20, 2, 2)

    assert sorted(path.name for path in destination.glob("*.tar")) == tar_names


def test_materializer_rejects_path_traversal(tmp_path):
    archive_root = tmp_path / "archive"
    output = tmp_path / "output"
    archive_root.mkdir()
    shard = archive_root / "metadata-00000.tar"
    with tarfile.open(shard, "w") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 0
        archive.addfile(info)
    manifest = {
        "complete": True,
        "components": {
            "metadata": {
                "shards": [
                    {"name": shard.name, "sha256": materialize.sha256_file(shard)}
                ]
            }
        },
    }
    (archive_root / "archive_manifest.json").write_text(json.dumps(manifest))

    try:
        materialize.materialize(archive_root, output, 1, verify_sha256=True)
    except RuntimeError as error:
        assert "Path traversal" in str(error)
    else:
        raise AssertionError("path traversal should have been rejected")


def test_smoke_selection_keeps_metadata_and_one_shard_per_split():
    manifest = {
        "components": {
            component: {
                "shards": [
                    {"name": f"{component}-00000.tar", "sha256": "first"},
                    {"name": f"{component}-00001.tar", "sha256": "second"},
                ]
            }
            for component in ("metadata", "train", "validation", "test")
        }
    }

    selected = materialize.selected_shards(manifest, 1)

    assert set(selected) == {
        "metadata-00000.tar",
        "metadata-00001.tar",
        "train-00000.tar",
        "validation-00000.tar",
        "test-00000.tar",
    }


def test_training_spec_streams_archive_and_frees_local_tar_copies():
    args = launcher.parse_args(
        [
            "--archive-volume",
            "prism-archive",
            "--result-volume",
            "prism-results",
            "--git-commit",
            "a" * 40,
        ]
    )
    spec = launcher.build_training_spec(args)

    assert "import" not in spec
    assert spec["resources"]["preset"] == "a100-1"
    command = spec["run"][0]["command"]
    assert "PRISM_ARCHIVE_VOLUME=prism-archive" in command
    assert "PRISM_DELETE_ARCHIVES_AFTER_EXTRACT=true" in command
    assert "PRISM_CHECKPOINT_URI=volume://vessl-storage/prism-results" in command
    assert "git fetch --depth 1 origin " + "a" * 40 in command


def test_mock_training_spec_uses_one_shard_and_one_epoch_per_stage():
    args = launcher.parse_args(
        [
            "--archive-volume",
            "prism-archive",
            "--result-volume",
            "prism-results-smoke",
            "--git-commit",
            "b" * 40,
            "--mock",
        ]
    )
    command = launcher.build_training_spec(args)["run"][0]["command"]

    assert "PRISM_ARCHIVE_MAX_SHARDS_PER_COMPONENT=1" in command
    assert "PRISM_STAGE4_EPOCHS=1" in command
    assert "PRISM_DIFFUSION_STEPS=2" in command


def test_wrapped_upload_error_is_recognized_as_expired_credentials():
    error = RuntimeError(
        "Failed to upload train-00004.tar: An error occurred (ExpiredToken) "
        "when calling the CreateMultipartUpload operation"
    )

    assert repack.VesslObjectStore._credential_error(error)


def test_read_json_returns_none_when_key_is_missing_after_token_refresh():
    class FakeClientError(Exception):
        def __init__(self, code):
            self.response = {"Error": {"Code": code}}
            super().__init__(code)

    class FakeClient:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)

        def get_object(self, **_kwargs):
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    store = object.__new__(repack.VesslObjectStore)
    store._client_error = FakeClientError
    store.destination_prefix = "archive"
    store.destination_bucket = "bucket"
    store.destination_client = FakeClient([FakeClientError("ExpiredToken")])
    refreshed_client = FakeClient([FakeClientError("NoSuchKey")])
    refresh_count = 0

    def refresh_destination():
        nonlocal refresh_count
        refresh_count += 1
        store.destination_client = refreshed_client

    store._refresh_destination = refresh_destination

    assert store.read_json("archive_manifest.json") is None
    assert refresh_count == 1


def test_read_json_succeeds_after_token_refresh():
    class FakeClientError(Exception):
        def __init__(self, code):
            self.response = {"Error": {"Code": code}}
            super().__init__(code)

    class FakeClient:
        def __init__(self, outcomes):
            self.outcomes = list(outcomes)

        def get_object(self, **_kwargs):
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    store = object.__new__(repack.VesslObjectStore)
    store._client_error = FakeClientError
    store.destination_prefix = "archive"
    store.destination_bucket = "bucket"
    store.destination_client = FakeClient([FakeClientError("ExpiredToken")])
    refreshed_client = FakeClient(
        [{"Body": io.BytesIO(b'{"complete": true}')}]
    )

    def refresh_destination():
        store.destination_client = refreshed_client

    store._refresh_destination = refresh_destination

    assert store.read_json("archive_manifest.json") == {"complete": True}
