"""
Fix notebook paths so they work in VS Code regardless of
which directory the kernel starts from.

Run from project root: python scripts/fix_paths.py
"""
import json
import os

SETUP_PATCH = "\n".join([
    "# Path setup: works whether VS Code launches from project root or notebooks/",
    "import os",
    "from pathlib import Path",
    "_cwd = Path(os.getcwd())",
    "PROJECT_ROOT = _cwd.parent if _cwd.name == 'notebooks' else _cwd",
    "os.chdir(PROJECT_ROOT)",
    'print(f"Working directory set to: {PROJECT_ROOT}")',
])

PATCH_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": SETUP_PATCH,
    "id": "path-setup-00"
}

NB_FILES = [
    "notebooks/01_eda_revenue_segmentation.ipynb",
    "notebooks/02_pricing_drivers.ipynb",
    "notebooks/03_host_strategy_location.ipynb",
]

def fix_paths(source):
    """Replace relative paths that assume cwd=notebooks/ with project-root-relative paths."""
    if isinstance(source, list):
        return [fix_paths(s) for s in source]
    return (
        source
        .replace("../data/raw/", "data/raw/")
        .replace("../outputs/figures", "outputs/figures")
    )

for nb_file in NB_FILES:
    with open(nb_file, encoding="utf-8") as f:
        nb = json.load(f)

    # Fix all paths in code cells
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = fix_paths(cell["source"])

    # Check if path-setup cell already inserted (avoid duplicates)
    already_patched = any(
        "PROJECT_ROOT" in "".join(c["source"]) if isinstance(c["source"], list) else "PROJECT_ROOT" in c["source"]
        for c in nb["cells"] if c["cell_type"] == "code"
    )

    if not already_patched:
        # Insert path-setup cell before the first code cell
        first_code_idx = next(
            i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
        )
        patch = dict(PATCH_CELL)
        patch["id"] = "path-setup-" + str(first_code_idx)
        nb["cells"].insert(first_code_idx, patch)

    with open(nb_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Patched: {nb_file}")

print("\nAll notebooks updated. Re-run them in VS Code.")
