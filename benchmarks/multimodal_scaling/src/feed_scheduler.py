"""Feed benchmark arrays without exceeding the site's per-user submit limit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

from common import HARNESS_ROOT, ROOT


def table_size(protocol: str, device: str) -> int:
    path = ROOT / f"metadata/tasks-{protocol}-{device}.tsv"
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle, delimiter="\t"))


def active_jobs() -> int:
    output = subprocess.check_output(
        ["squeue", "-r", "-u", os.environ["USER"], "-h", "-o", "%i"],
        text=True,
    )
    return len(output.splitlines())


def persist(
    path: Path, targets: list[dict], job_ids: list[str], complete: bool = False
) -> None:
    payload = {
        "complete": complete,
        "targets": targets,
        "submitted_job_ids": job_ids,
        "updated_unix_time": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def submit_array(
    protocol: str, device: str, offset: int, length: int, concurrency: int
):
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
    script = HARNESS_ROOT / "slurm" / f"run_{device}_array.sh"
    return (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                f"--array=0-{length - 1}%{concurrency}",
                f"--export={export}",
                str(script),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        .strip()
        .split(";", 1)[0]
    )


def submit_finalizer(job_ids: list[str]) -> str:
    dependency = f"afterany:{':'.join(job_ids)}"
    export = ",".join(
        (
            "ALL",
            f"TMOL_BENCH_ROOT={ROOT}",
            f"TMOL_BENCH_HARNESS={HARNESS_ROOT}",
        )
    )
    return (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                f"--dependency={dependency}",
                f"--export={export}",
                str(HARNESS_ROOT / "slurm" / "plot_existing.sh"),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        .strip()
        .split(";", 1)[0]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        action="append",
        metavar="PROTOCOL:DEVICE:OFFSET",
        help="Ordered table segment to submit; repeat for subsequent segments.",
    )
    parser.add_argument(
        "--resume-state",
        action="store_true",
        help="Resume offsets and dependency IDs from --state after a requeue.",
    )
    parser.add_argument("--dependency-job", action="append", default=[])
    parser.add_argument("--job-limit", type=int, default=3000)
    parser.add_argument("--reserve", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--cpu-concurrency", type=int, default=16)
    parser.add_argument("--cuda-concurrency", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--state",
        type=Path,
        default=ROOT / "metadata/scheduler_feeder_state.json",
    )
    args = parser.parse_args()

    if args.resume_state and args.state.exists():
        previous = json.loads(args.state.read_text())
        if previous.get("complete"):
            print("feeder state is already complete", flush=True)
            return
        targets = previous["targets"]
        job_ids = list(previous["submitted_job_ids"])
        print(f"resumed {args.state}", flush=True)
    else:
        if not args.target:
            raise ValueError("at least one --target is required for a new feeder")
        targets = []
        for value in args.target:
            protocol, device, offset_text = value.split(":")
            if protocol not in {"score_gradient", "fastrelax"}:
                raise ValueError(f"unknown protocol: {protocol}")
            if device not in {"cpu", "cuda"}:
                raise ValueError(f"unknown device: {device}")
            offset = int(offset_text)
            count = table_size(protocol, device)
            if not 0 <= offset <= count:
                raise ValueError(f"offset {offset} outside {protocol}/{device} table")
            targets.append(
                {
                    "protocol": protocol,
                    "device": device,
                    "offset": offset,
                    "count": count,
                }
            )
        job_ids = list(args.dependency_job)
    persist(args.state, targets, job_ids)
    for target in targets:
        while target["offset"] < target["count"]:
            capacity = args.job_limit - args.reserve - active_jobs()
            if capacity <= 0:
                time.sleep(args.poll_seconds)
                continue
            length = min(args.chunk_size, capacity, target["count"] - target["offset"])
            concurrency = (
                args.cpu_concurrency
                if target["device"] == "cpu"
                else args.cuda_concurrency
            )
            try:
                job_id = submit_array(
                    target["protocol"],
                    target["device"],
                    target["offset"],
                    length,
                    concurrency,
                )
            except subprocess.CalledProcessError as error:
                print(error.output.strip(), flush=True)
                time.sleep(args.poll_seconds)
                continue
            print(
                target["protocol"],
                target["device"],
                target["offset"],
                length,
                job_id,
                flush=True,
            )
            target["offset"] += length
            job_ids.append(job_id)
            persist(args.state, targets, job_ids)

    while True:
        try:
            finalizer = submit_finalizer(job_ids)
            break
        except subprocess.CalledProcessError as error:
            print(error.output.strip(), flush=True)
            time.sleep(args.poll_seconds)
    print("finalizer", finalizer, flush=True)
    persist(args.state, targets, [*job_ids, finalizer], complete=True)


if __name__ == "__main__":
    main()
