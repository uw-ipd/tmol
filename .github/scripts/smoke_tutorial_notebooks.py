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
TUTORIAL_REF = "kdidi/sphinx-docs"


def _validate_tutorial_entrypoints(notebook, path: Path) -> None:
    """Require every published tutorial to expose Colab and local setup."""
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    if "Open In Colab" not in markdown:
        raise ValueError(f"{path} is missing its Open In Colab badge")
    if "setup_colab(" not in code:
        raise ValueError(f"{path} is missing its guarded Colab bootstrap")
    if notebook.metadata.get("accelerator") != "GPU":
        raise ValueError(f"{path} does not request a Colab GPU runtime")
    if f"blob/{TUTORIAL_REF}/docs/tutorial/" not in markdown:
        raise ValueError(f"{path} does not use the current tutorial-branch Colab URL")
    if f"{TUTORIAL_REF}/docs/tutorial/colab_setup.py" not in code:
        raise ValueError(f"{path} does not use the current tutorial bootstrap")


def _without_gpu_cells(notebook):
    """Copy a notebook and replace ``gpu-only`` cells with a CPU skip."""
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
            cell.source = (
                "print('Skipped by CPU documentation build: this cell requires CUDA.')"
            )
            cell.outputs = []
            cell.execution_count = None
    return notebook


def execute_notebook(path: Path, timeout: int, *, write: bool = False) -> None:
    """Execute one notebook, optionally retaining outputs for Sphinx."""
    with path.open(encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    _validate_tutorial_entrypoints(notebook, path)
    executed = _without_gpu_cells(notebook)
    client = NotebookClient(
        executed,
        timeout=timeout,
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent.resolve())}},
    )
    client.execute()
    if write:
        # Preserve the authored source (including GPU examples), but copy the
        # CPU execution results used by nbsphinx into the CI workspace.
        for source_cell, executed_cell in zip(notebook.cells, executed.cells):
            if source_cell.cell_type != "code":
                continue
            source_cell.outputs = executed_cell.outputs
            source_cell.execution_count = executed_cell.execution_count
        with path.open("w", encoding="utf-8") as handle:
            nbformat.write(notebook, handle)


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
    parser.add_argument(
        "--write",
        action="store_true",
        help="Retain executed outputs in the source notebooks for a docs build",
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
        execute_notebook(path, args.timeout, write=args.write)
    print(f"Successfully executed {len(paths)} tutorial notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
