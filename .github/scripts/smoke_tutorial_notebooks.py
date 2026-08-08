#!/usr/bin/env python3
"""Execute tutorial notebooks in memory without modifying checked-in files."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


PLANNED_NOTEBOOKS = (
    "01_working_with_tmol.ipynb",
    "02_gpu_batching.ipynb",
    "03_scoring_and_analysis.ipynb",
    "04_packing_and_mutation_scan.ipynb",
    "05_minimization_constraints_kinematics.ipynb",
    "06_fast_relax.ipynb",
    "07_ligand_and_params.ipynb",
    "08_nucleic_acids.ipynb",
)


def _without_gpu_cells(notebook):
    """Copy a notebook and replace ``gpu-only`` cells with no-op code."""
    notebook = copy.deepcopy(notebook)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        has_source_tag = any(
            line.strip() == "#| tags: [gpu-only]"
            for line in cell.source.splitlines()[:3]
        )
        if "gpu-only" in tags or has_source_tag:
            cell.source = "# Skipped by CPU notebook smoke check: gpu-only"
            cell.outputs = []
            cell.execution_count = None
    return notebook


def execute_notebook(path: Path, timeout: int) -> None:
    """Execute one notebook in memory and raise on cell errors."""
    with path.open(encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    notebook = _without_gpu_cells(notebook)
    client = NotebookClient(
        notebook,
        timeout=timeout,
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent.resolve())}},
    )
    client.execute()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="Notebook paths (defaults to the eight planned tutorials when present)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-cell timeout in seconds (default: 600)",
    )
    args = parser.parse_args()

    # Ensure kernels spawned by nbclient cannot see a runner GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.notebooks:
        paths = args.notebooks
    else:
        tutorial_dir = Path("docs/tutorial")
        paths = [
            tutorial_dir / name
            for name in PLANNED_NOTEBOOKS
            if (tutorial_dir / name).is_file()
        ]

    if not paths:
        print("No planned tutorial notebooks are present; nothing to execute.")
        return 0

    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error("Notebook not found: " + ", ".join(str(path) for path in missing))

    for path in paths:
        print(f"Executing {path}")
        execute_notebook(path, args.timeout)
    print(f"Successfully executed {len(paths)} tutorial notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
