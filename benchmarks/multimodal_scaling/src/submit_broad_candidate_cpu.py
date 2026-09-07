"""Freeze provenance and submit the all-input candidate CPU arrays."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from common import HARNESS_ROOT, ROOT


def command(*parts: str) -> str:
    return subprocess.check_output(parts, text=True).strip()


def task_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--candidate-env", type=Path, required=True)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        default=Path("/mnt/home/kdidi/tmol-paper-sources/v0.1.55"),
    )
    parser.add_argument(
        "--baseline-env",
        type=Path,
        default=Path("/mnt/home/kdidi/tmol-paper-benchmark/envs/tmol-0.1.55"),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path(
            "/mnt/home/kdidi/apptainer-artifacts/"
            "latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif"
        ),
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("score_gradient", "fastrelax"),
        default=("score_gradient", "fastrelax"),
    )
    parser.add_argument("--chunk-size", type=int, default=384)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 1000:
        parser.error("chunk size must be between 1 and the site's 1,000-task limit")

    output = args.output_root.resolve()
    metadata = output / "metadata"
    logs = output / "logs"
    metadata.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(HARNESS_ROOT / "src/build_broad_candidate_cpu_tasks.py"),
            "--output",
            str(metadata),
        ],
        check=True,
        env={**os.environ, "TMOL_BENCH_ROOT": str(ROOT)},
    )

    candidate_commit = command(
        "git", "-C", str(args.candidate_source), "rev-parse", "HEAD"
    )
    if not args.dry_run:
        reference_files = [
            ROOT / "metadata/benchmark_spec.json",
            ROOT / "metadata/dataset_manifest.csv",
            ROOT / "results/summary/timing_summary.csv",
            *(metadata / f"tasks-broad-candidate-{p}-cpu.tsv" for p in args.protocols),
        ]
        provenance = metadata / "broad_candidate_cpu_provenance.json"
        provenance_command = [
            sys.executable,
            str(HARNESS_ROOT / "src/capture_candidate_provenance.py"),
            "--baseline-source",
            str(args.baseline_source),
            "--baseline-env",
            str(args.baseline_env),
            "--candidate-source",
            str(args.candidate_source),
            "--candidate-env",
            str(args.candidate_env),
            "--harness-root",
            str(HARNESS_ROOT.parents[1]),
            "--image",
            str(args.image),
            "--output",
            str(provenance),
        ]
        for path in reference_files:
            provenance_command.extend(("--reference-file", str(path)))
        subprocess.run(provenance_command, check=True)

    script = HARNESS_ROOT / "slurm/run_broad_candidate_cpu_array.sh"
    for protocol in args.protocols:
        table = metadata / f"tasks-broad-candidate-{protocol}-cpu.tsv"
        count = task_count(table)
        if not 0 <= args.start_offset <= count:
            parser.error(f"start offset {args.start_offset} is outside {count} tasks")
        for offset in range(args.start_offset, count, args.chunk_size):
            length = min(args.chunk_size, count - offset)
            export = ",".join(
                (
                    "ALL",
                    f"TMOL_BENCH_ROOT={ROOT}",
                    f"TMOL_BENCH_HARNESS={HARNESS_ROOT}",
                    f"TMOL_BROAD_CANDIDATE_ROOT={output}",
                    f"TMOL_CANDIDATE_SOURCE={args.candidate_source.resolve()}",
                    f"TMOL_CANDIDATE_ENV={args.candidate_env.resolve()}",
                    f"TMOL_CANDIDATE_COMMIT={candidate_commit}",
                    f"TMOL_BENCH_IMAGE={args.image.resolve()}",
                    f"TMOL_BENCH_PROTOCOL={protocol}",
                    f"TMOL_TASK_OFFSET={offset}",
                )
            )
            submission = [
                "sbatch",
                "--parsable",
                f"--array=0-{length - 1}%{args.concurrency}",
                f"--output={logs}/{protocol}-%A_%a.out",
                f"--error={logs}/{protocol}-%A_%a.err",
                f"--export={export}",
                str(script),
            ]
            if args.dry_run:
                print(" ".join(submission))
            else:
                print(protocol, offset, length, command(*submission))


if __name__ == "__main__":
    main()
