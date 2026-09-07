"""Submit task tables in Slurm-array chunks below the site array-size limit."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

from common import HARNESS_ROOT, ROOT


def task_count(protocol: str, device: str) -> int:
    path = ROOT / f"metadata/tasks-{protocol}-{device}.tsv"
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=("score_gradient", "fastrelax"),
        required=True,
    )
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--cpu-concurrency", type=int, default=16)
    parser.add_argument("--cuda-concurrency", type=int, default=8)
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=("cpu", "cuda"),
        default=("cpu", "cuda"),
        help="Submit only the selected task tables.",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Resume at this zero-based row in each selected task table.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.chunk_size <= 1000:
        raise ValueError("chunk size must fit the site's MaxArraySize=1001")
    for protocol in args.protocols:
        for device in args.devices:
            count = task_count(protocol, device)
            if not 0 <= args.start_offset <= count:
                raise ValueError(
                    f"start offset {args.start_offset} is outside the {count}-row "
                    f"{protocol}/{device} table"
                )
            concurrency = (
                args.cpu_concurrency if device == "cpu" else args.cuda_concurrency
            )
            script = HARNESS_ROOT / "slurm" / f"run_{device}_array.sh"
            for offset in range(args.start_offset, count, args.chunk_size):
                length = min(args.chunk_size, count - offset)
                export = ",".join(
                    (
                        "ALL",
                        f"TMOL_BENCH_ROOT={ROOT}",
                        f"TMOL_BENCH_HARNESS={HARNESS_ROOT}",
                        f"TMOL_BENCH_PROTOCOL={protocol}",
                        f"TMOL_BENCH_DEVICE={device}",
                        f"TMOL_TASK_OFFSET={offset}",
                    )
                )
                command = [
                    "sbatch",
                    "--parsable",
                    f"--array=0-{length - 1}%{concurrency}",
                    f"--export={export}",
                    str(script),
                ]
                try:
                    job_id = (
                        "DRY_RUN"
                        if args.dry_run
                        else subprocess.check_output(
                            command,
                            text=True,
                            env=os.environ,
                            stderr=subprocess.STDOUT,
                        ).strip()
                    )
                except subprocess.CalledProcessError as error:
                    print(
                        f"submission stopped at {protocol}/{device} offset {offset}: "
                        f"{error.output.strip()}",
                        file=sys.stderr,
                    )
                    print(
                        "Resume after scheduler capacity is available with: "
                        f"--protocols {protocol} --devices {device} "
                        f"--start-offset {offset}",
                        file=sys.stderr,
                    )
                    raise SystemExit(error.returncode) from None
                print(protocol, device, offset, length, job_id)


if __name__ == "__main__":
    main()
