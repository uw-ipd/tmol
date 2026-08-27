#!/usr/bin/env python3
"""Run the supported TMol workflow benchmark or Nsight Systems matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "dev" / "profiling" / "workflows.py"

# Small, medium, and large proteins plus non-protein chemistry. Batch sizes are
# intentionally capped so the complete suite is practical on one H200.
SCORE_CASES = [
    (system, batch)
    for system, batches in {
        "protein015": (1, 32),
        "protein100": (1, 16),
        "protein400": (1, 4),
        "dna24": (1, 32),
        "rna33": (1, 32),
        "protein_dna": (1, 8),
        "protein_ligand": (1, 8),
    }.items()
    for batch in batches
]

MATRIX = {
    workflow: SCORE_CASES
    for workflow in ("score", "score_grad", "score_graph", "score_graph_grad")
}
MATRIX.update(
    {
        "cart_min": [
            ("protein015", 1),
            ("protein015", 32),
            ("protein100", 1),
            ("protein100", 8),
            ("protein400", 1),
            ("dna24", 1),
            ("dna24", 8),
            ("rna33", 1),
            ("rna33", 8),
            ("protein_dna", 1),
            ("protein_dna", 4),
            ("protein_ligand", 1),
            ("protein_ligand", 4),
        ],
        "kin_min": [
            ("protein015", 1),
            ("protein015", 16),
            ("protein100", 1),
            ("protein100", 4),
            ("dna24", 1),
            ("dna24", 4),
            ("rna33", 1),
            ("rna33", 4),
            ("protein_dna", 1),
            ("protein_dna", 2),
        ],
        "fast_relax": [
            ("protein015", 1),
            ("protein015", 8),
            ("protein100", 1),
            ("protein100", 4),
        ],
    }
)


def _defaults(workflow: str, trace: bool) -> tuple[int, int]:
    if trace:
        return (1, 1)
    if workflow.startswith("score"):
        return (5, 30)
    if workflow == "fast_relax":
        return (1, 3)
    return (1, 5)


def _run(command: list[str], log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(
            command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT
        )


def _write_summary(result_dir: Path):
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        data = json.loads(path.read_text())
        rows.append(
            {
                key: data[key]
                for key in (
                    "case",
                    "workflow",
                    "system",
                    "batch",
                    "poses",
                    "system_setup_ms",
                    "workload_setup_ms",
                    "median_ms",
                    "mean_ms",
                    "min_ms",
                    "p95_ms",
                    "memory_before_bytes",
                    "peak_memory_bytes",
                    "peak_memory_delta_bytes",
                )
            }
        )
    if not rows:
        return
    with (result_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output_dir: Path, cases, trace: bool):
    files = [
        {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    manifest = {
        "runner_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "runner_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        ),
        "python": platform.python_version(),
        "nsys": (
            subprocess.run(
                ["nsys", "--version"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if trace
            else None
        ),
        "trace": trace,
        "cases": [f"{workflow}-{system}-b{batch}" for workflow, system, batch in cases],
        "files": files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--graph-node-trace",
        action="store_true",
        help="trace individual CUDA graph nodes (high overhead)",
    )
    parser.add_argument("--workflow", choices=MATRIX, action="append")
    parser.add_argument("--case", action="append", help="exact case slug to select")
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="replace completed cases")
    args = parser.parse_args()

    workflows = args.workflow or list(MATRIX)
    cases = [
        (workflow, system, batch)
        for workflow in workflows
        for system, batch in MATRIX[workflow]
        if not args.case or f"{workflow}-{system}-b{batch}" in args.case
    ]
    if args.case and len(cases) != len(set(args.case)):
        parser.error("one or more --case values are not in the supported matrix")

    result_dir = args.output_dir / "results"
    trace_dir = args.output_dir / "traces"
    log_dir = args.output_dir / "logs"
    for workflow, system, batch in cases:
        case = f"{workflow}-{system}-b{batch}"
        warmup, iterations = _defaults(workflow, args.trace)
        output = result_dir / f"{case}.json"
        complete = [output]
        if args.trace:
            complete.extend(
                [
                    trace_dir / f"{case}.nsys-rep",
                    trace_dir / f"{case}.sqlite",
                    trace_dir / f"{case}.stats.csv",
                ]
            )
        if not args.force and all(path.is_file() for path in complete):
            print(f"[skip] {case}", flush=True)
            continue
        command = [
            sys.executable,
            str(RUNNER),
            "--workflow",
            workflow,
            "--system",
            system,
            "--batch",
            str(batch),
            "--max-iter",
            str(args.max_iter),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--output",
            str(output),
        ]
        if args.trace:
            trace_dir.mkdir(parents=True, exist_ok=True)
            command += ["--capture"]
            command = [
                "nsys",
                "profile",
                "--force-overwrite=true",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--trace=cuda,nvtx,osrt",
                *(["--cuda-graph-trace=node"] if args.graph_node_trace else []),
                "--sample=none",
                "--cpuctxsw=none",
                "--output",
                str(trace_dir / case),
                *command,
            ]
        print(
            f"[{cases.index((workflow, system, batch)) + 1}/{len(cases)}] {case}",
            flush=True,
        )
        _run(command, log_dir / f"{case}.log")
        if args.trace:
            report = trace_dir / f"{case}.nsys-rep"
            _run(
                [
                    "nsys",
                    "stats",
                    "--force-export=true",
                    "--report",
                    "cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum",
                    "--format",
                    "csv",
                    str(report),
                ],
                trace_dir / f"{case}.stats.csv",
            )

    _write_summary(result_dir)
    _write_manifest(args.output_dir, cases, args.trace)


if __name__ == "__main__":
    main()
