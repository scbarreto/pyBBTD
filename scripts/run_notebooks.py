"""Execute the documentation notebooks.

By default every notebook under docs/ is run and saved with its outputs, which
the documentation shows as they are. With --check, the notebooks are only run,
to confirm they still work, and the files are left untouched.

Warnings printed by a cell start with the path of a temporary file of the
kernel; that machine-specific prefix is removed from the saved outputs.

    uv run --extra notebooks python scripts/run_notebooks.py [--check] [paths ...]
"""

import argparse
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

DOCS = Path(__file__).resolve().parents[1] / "docs"
KERNEL_FILE_PREFIX = re.compile(r"\S*ipykernel_\d+/\d+\.py:\d+: ")


def strip_kernel_paths(nb):
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.get("name") == "stderr":
                output["text"] = KERNEL_FILE_PREFIX.sub("", output["text"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, help="notebooks to run")
    parser.add_argument("--check", action="store_true", help="run without saving")
    args = parser.parse_args()

    paths = args.paths or sorted(
        p for p in DOCS.rglob("*.ipynb") if "_build" not in p.parts
    )
    for path in paths:
        start = time.time()
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(
            nb,
            timeout=1200,
            kernel_name="python3",
            resources={"metadata": {"path": path.parent}},
        )
        client.execute()
        if not args.check:
            strip_kernel_paths(nb)
            nbformat.write(nb, path)
        print(f"{path.relative_to(DOCS.parent)}: ran in {time.time() - start:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
