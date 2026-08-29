#!/usr/bin/env python3
"""Validate and execute the published tutorial notebooks."""

from __future__ import annotations

import argparse
import copy
import os
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

TUTORIAL_NOTEBOOKS = (
    "01_working_with_tmol.ipynb",
    "02_gpu_batching.ipynb",
    "03_scoring_and_analysis.ipynb",
    "04_packing_and_mutation_scan.ipynb",
    "05_minimization_constraints_kinematics.ipynb",
    "06_fast_relax.ipynb",
    "07_ligand_and_params.ipynb",
    "08_nucleic_acids.ipynb",
    "09_protein_interface_hotspot_scan.ipynb",
)
TUTORIAL_REF = "master"
KERNEL_STARTUP_TIMEOUT = 180
CPU_CUDA_RUNTIME_NOISE = re.compile(
    r"^W\d{4} .* torch/utils/cpp_extension\.py:\d+\] "
    r"No CUDA runtime is found, using CUDA_HOME='[^']*'\n?$"
)


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
        raise ValueError(f"{path} does not use the durable Colab URL")
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
                "print('Skipped by CPU smoke execution: this cell requires CUDA.')"
            )
            cell.outputs = []
            cell.execution_count = None
    return notebook


def _remove_expected_cpu_cuda_noise(notebook) -> None:
    """Remove PyTorch's expected no-CUDA warning from published CPU outputs."""
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        retained = []
        for output in cell.outputs:
            if output.output_type != "stream":
                retained.append(output)
                continue
            text = "".join(
                line
                for line in output.text.splitlines(keepends=True)
                if not CPU_CUDA_RUNTIME_NOISE.fullmatch(line)
            )
            if text:
                output.text = text
                retained.append(output)
        cell.outputs = retained


def execute_notebook(
    path: Path,
    timeout: int,
    *,
    write: bool = False,
    include_gpu_cells: bool = False,
) -> None:
    """Execute one notebook, optionally retaining outputs for Sphinx."""
    with path.open(encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    _validate_tutorial_entrypoints(notebook, path)
    executed = (
        copy.deepcopy(notebook) if include_gpu_cells else _without_gpu_cells(notebook)
    )
    client = NotebookClient(
        executed,
        timeout=timeout,
        startup_timeout=KERNEL_STARTUP_TIMEOUT,
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent.resolve())}},
    )
    client.execute()
    if not include_gpu_cells:
        _remove_expected_cpu_cuda_noise(executed)
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
        help="Notebook paths (defaults to the nine published tutorials when present)",
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
    parser.add_argument(
        "--execution-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "Execution device policy. CPU hides CUDA and replaces gpu-only cells; "
            "CUDA requires a visible GPU and executes every cell (default: cpu)."
        ),
    )
    args = parser.parse_args()

    include_gpu_cells = args.execution_device == "cuda"
    if include_gpu_cells:
        import torch

        if not torch.cuda.is_available():
            parser.error(
                "--execution-device cuda requires a CUDA device visible to PyTorch"
            )
    else:
        # Ensure kernels spawned by nbclient cannot see a runner GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if args.notebooks:
        paths = args.notebooks
    else:
        tutorial_dir = Path("docs/tutorial")
        paths = [
            tutorial_dir / name
            for name in TUTORIAL_NOTEBOOKS
            if (tutorial_dir / name).is_file()
        ]

    if not paths:
        print("No tutorial notebooks are present; nothing to execute.")
        return 0

    missing = [path for path in paths if not path.is_file()]
    if missing:
        parser.error("Notebook not found: " + ", ".join(str(path) for path in missing))

    for path in paths:
        print(f"Executing {path}")
        execute_notebook(
            path,
            args.timeout,
            write=args.write,
            include_gpu_cells=include_gpu_cells,
        )
    print(f"Successfully executed {len(paths)} tutorial notebook(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
