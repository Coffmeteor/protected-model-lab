from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import folder_paths

folder_paths.add_model_folder_path("protected_carriers", str(PROJECT_ROOT / "outputs/carriers"))
folder_paths.add_model_folder_path("protected_cores", str(PROJECT_ROOT / "outputs/cores"))

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
