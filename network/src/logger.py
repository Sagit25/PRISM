from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class WandbLogger:
    """Small optional W&B wrapper.

    Importing the PRISM package must not require W&B.  Training can be run
    fully offline with ``mode="disabled"`` and only imports W&B when logging is
    requested.
    """

    def __init__(
        self,
        project: str,
        name: str,
        config: dict[str, Any],
        *,
        mode: str = "disabled",
        entity: str | None = None,
        group: str | None = None,
        tags: Sequence[str] = (),
        notes: str | None = None,
    ) -> None:
        self._run = None
        self._wandb = None
        if mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "W&B logging requires refractive-mam2[experiment]"
            ) from exc
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=name,
            config=config,
            mode=mode,
            entity=entity,
            group=group,
            tags=list(tags),
            notes=notes,
        )
        # Keep every panel and scalar aligned on the optimizer-step axis.
        wandb.define_metric("global_step")
        for namespace in ("train/*", "validation/*", "test/*"):
            wandb.define_metric(namespace, step_metric="global_step")

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(
        self,
        metrics: dict[str, Any],
        step: int,
        *,
        commit: bool = True,
    ) -> None:
        if self._run is None:
            return
        self._wandb.log(
            {
                "global_step": step,
                **{
                    key: (
                        value.detach().item()
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in metrics.items()
                },
            },
            commit=commit,
        )

    def log_images(
        self,
        key: str,
        images: Sequence[Any],
        step: int,
        *,
        captions: Sequence[str] | None = None,
        commit: bool = True,
    ) -> None:
        """Log PIL/NumPy/Tensor images without importing W&B when disabled."""

        if self._run is None or not images:
            return
        if captions is None:
            captions = [""] * len(images)
        if len(images) != len(captions):
            raise ValueError("images and captions must have the same length")
        payload = [
            self._wandb.Image(image, caption=caption)
            for image, caption in zip(images, captions)
        ]
        self._wandb.log(
            {"global_step": step, key: payload},
            commit=commit,
        )

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
