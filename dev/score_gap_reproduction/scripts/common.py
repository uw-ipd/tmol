from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "metadata/benchmark_spec.json").read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest() -> list[dict[str, str]]:
    rows = []
    for name in ("dataset_manifest.csv", "chemistry_manifest.csv"):
        path = ROOT / "metadata" / name
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for key in ("structure_path", "ligand_tmol_params", "ligand_rosetta_params"):
                    if row.get(key):
                        row[key] = str(ROOT / row[key])
                if row.get("status") == "ok":
                    assert sha256(Path(row["structure_path"])) == row["structure_sha256"]
                rows.append(row)
    return rows


def write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def append_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    incoming = list(rows)
    if not incoming:
        return
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    write_rows(path, [*existing, *incoming])


def machine_metadata() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                args, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    lscpu = command("lscpu") or ""
    cpu_model = next(
        (
            line.split(":", 1)[1].strip()
            for line in lscpu.splitlines()
            if line.startswith("Model name:")
        ),
        platform.processor() or None,
    )
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu_model,
        "gpu": command(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def synchronize(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def benchmark_callable(
    function: Callable[[], Any],
    *,
    device: str,
    warmups: int | None = None,
    samples: int | None = None,
) -> tuple[list[float], int]:
    """Return seconds per call and the adaptive inner-loop iteration count."""
    warmups = SPEC["warmup_iterations"] if warmups is None else warmups
    samples = SPEC["timed_samples"] if samples is None else samples
    for _ in range(warmups):
        function()
    synchronize(device)
    start = time.perf_counter()
    function()
    synchronize(device)
    estimate = max(time.perf_counter() - start, 1e-9)
    iterations = round(SPEC["target_seconds_per_sample"] / estimate)
    iterations = max(SPEC["minimum_iterations"], iterations)
    iterations = min(SPEC["maximum_iterations"], iterations)

    timings = []
    for _ in range(samples):
        synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        synchronize(device)
        timings.append((time.perf_counter() - start) / iterations)
    return timings, iterations
