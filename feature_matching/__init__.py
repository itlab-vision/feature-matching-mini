import sys
from pathlib import Path

def _init_thirdparty_routes():
    root_dir = Path(__file__).resolve().parent.parent
    thirdparty_dir = root_dir / "thirdparty"
    
    if thirdparty_dir.exists():
        for item in thirdparty_dir.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                resolved_path = str(item.resolve())
                if resolved_path not in sys.path:
                    sys.path.insert(0, resolved_path)

_init_thirdparty_routes()
