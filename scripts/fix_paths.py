"""
Fix notebook path-setup cells.
Run from project root: python scripts/fix_paths.py
"""
import json

NEW_SETUP = "\n".join([
    "import os",
    "from pathlib import Path",
    "",
    "def _find_root():",
    "    p = Path(os.getcwd()).resolve()",
    "    for candidate in [p] + list(p.parents):",
    "        if (candidate / 'data' / 'raw').exists():",
    "            return candidate",
    "    raise FileNotFoundError('Cannot find project root — data/raw not found')",
    "",
    "PROJECT_ROOT = _find_root()",
    "os.chdir(PROJECT_ROOT)",
    "print('Project root:', PROJECT_ROOT)",
    "print('data/raw found:', (PROJECT_ROOT / 'data' / 'raw').exists())",
])

NB_FILES = [
    "notebooks/01_eda_revenue_segmentation.ipynb",
    "notebooks/02_pricing_drivers.ipynb",
    "notebooks/03_host_strategy_location.ipynb",
]

for nb_file in NB_FILES:
    with open(nb_file, encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
            if "PROJECT_ROOT" in src:
                cell["source"] = NEW_SETUP
                cell["outputs"] = []
                cell["execution_count"] = None
                break

    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Patched:", nb_file)
