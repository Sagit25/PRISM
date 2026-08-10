"""Camera pose implementations for single and multi-view rendering."""
import importlib
from pathlib import Path

# Auto-import all .py submodules to trigger @CAMPOSE_REGISTRY.register()
_pkg_dir = Path(__file__).parent
for _f in _pkg_dir.glob('*.py'):
    if _f.name != '__init__.py':
        importlib.import_module(f'.{_f.stem}', package=__name__)
