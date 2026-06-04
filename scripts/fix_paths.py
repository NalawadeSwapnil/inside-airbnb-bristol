"""
Fix notebook path-setup cells to robustly find the project root
regardless of where VS Code starts the kernel from.

Run from project root: python scripts/fix_paths.py
"""
import json
import os

# Robust path-setup: walks UP from cwd until it finds data/raw/
NEW_SETUP = "\n".join([
    "import os",
    "from pathlib import Path",
    "",
    "# Find project root by searching upward for the data/raw folder",
    "# Works regardless of where VS Code starts the kernel (project root, notebooks/, .venv, etc.)",
    "def _find_root():",
    "    p = Path(os.getcwd()).resolve()",
    "    for candidate in [p] + list(p.parents):",
    "        if (candidate / 'data' / 'raw').exists():",
    "            return candidate",
    "    raise FileNotFoundError('Could not find project root. Make sure data/raw/ exists.')",
    "",
    "PROJECT_ROOT = _find_root()",
    "os.chdir(PROJECT_ROOT)",
    'print(f"Project root: {PROJECT_ROOT}")',
    'print(f"data/raw exists: {(PROJECT_ROOT / \'data\' / \'raw\').exists()}")',
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
            # Replace the old path-setup cell with the new robust version
            if "PROJECT_ROOT" in src and "_find_root" not in src:
                cell["source"] = NEW_SETUP
                cell["outputs"] = []
                cell["execution_count"] = None
                print(f"  Updated path-setup cell in {nb_file}")
                break
    else:
        # No path-setup cell found at all — insert before first code cell
        first_code_idx = next(
            i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
        )
        nb["cells"].insert(first_code_idx, {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": NEW_SETUP,
            "id": "path-setup-robust"
        })
        print(f"  Inserted path-setup cell in {nb_file}")

    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nDone. Now in VS Code:")
print("1. Click the kernel (top-right) -> Select Another Kernel -> Jupyter Kernel")
print("2. Choose: StayPriceML (.venv)")
print("3. Ctrl+Shift+P -> Notebook: Restart Kernel and Run All Cells")
