import sys
from pathlib import Path


def _init_thirdparty_routes():
    project_root = Path(__file__).resolve().parent.parent
    thirdparty_dir = project_root / "thirdparty"

    if not thirdparty_dir.is_dir():
        return

    for item in thirdparty_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            paths = [item]
            paths.extend(subdir for subdir in item.iterdir() if subdir.is_dir() and not subdir.name.startswith("."))

            for path in paths:
                resolved_path = str(path.resolve())
                if resolved_path not in sys.path:
                    sys.path.insert(0, resolved_path)


_init_thirdparty_routes()