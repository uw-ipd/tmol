"""Capture immutable inputs and software revisions for a benchmark run."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path

from common import HARNESS_ROOT, ROOT, SPEC, sha256

DEFAULT_IMAGE = Path(
    "/mnt/home/kdidi/apptainer-artifacts/"
    "latent-dev-cuda13-26.06-tmol0.1.49-cueq0.10-full.sif"
)
DEFAULT_SOURCES = Path("/mnt/home/kdidi/tmol-paper-sources")
DEFAULT_ENVS = Path("/mnt/home/kdidi/tmol-paper-benchmark/envs")
PYROSETTA_SOURCE = Path("/mnt/home/kdidi/projects/pyrosetta-2024.39")


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state(path: Path) -> dict:
    revision = command("git", "-C", str(path), "rev-parse", "HEAD")
    status = command("git", "-C", str(path), "status", "--porcelain")
    return {
        "path": str(path),
        "revision": revision,
        "clean": status == "",
        "status": status,
    }


def checked_file(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def freeze(environment: Path) -> list[str] | None:
    output = command(str(environment / "bin/python"), "-m", "pip", "freeze", "--all")
    return output.splitlines() if output is not None else None


def main() -> None:
    image = Path(os.environ.get("TMOL_BENCH_IMAGE", DEFAULT_IMAGE))
    sources = Path(os.environ.get("TMOL_BENCH_SOURCES", DEFAULT_SOURCES))
    environments = Path(os.environ.get("TMOL_BENCH_ENVS", DEFAULT_ENVS))
    harness_git_root = Path(os.environ.get("TMOL_BENCH_HARNESS_GIT_ROOT", HARNESS_ROOT))
    versions = (
        SPEC["historical_tmol"]["default_version"],
        SPEC["historical_tmol"]["nucleic_version"],
        SPEC["latest_tmol"]["version"],
    )
    input_names = [
        "benchmark_spec.json",
        "dataset_manifest.csv",
        "tasks-score_gradient-cpu.tsv",
        "tasks-score_gradient-cuda.tsv",
        "tasks-fastrelax-cpu.tsv",
        "tasks-fastrelax-cuda.tsv",
    ]
    payload = {
        "schema_version": 1,
        "captured_unix_time": time.time(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "container": checked_file(image),
        "harness": git_state(harness_git_root),
        "tmol_sources": {
            version: git_state(sources / f"v{version}") for version in versions
        },
        "pyrosetta_source": {
            "path": str(PYROSETTA_SOURCE),
            "distribution": "pyrosetta-2024.39+release.59628fb",
            "files": {
                relative: checked_file(PYROSETTA_SOURCE / relative)
                for relative in (
                    "pyrosetta/rosetta.so",
                    "pyrosetta-2024.39+release.59628fb.dist-info/METADATA",
                    "pyrosetta-2024.39+release.59628fb.dist-info/RECORD",
                    "pyrosetta-2024.39+release.59628fb.dist-info/direct_url.json",
                )
            },
        },
        "python_environments": {
            version: {
                "path": str(environments / f"tmol-{version}"),
                "pip_freeze": freeze(environments / f"tmol-{version}"),
            }
            for version in versions
        },
        "inputs": {
            name: checked_file(ROOT / "metadata" / name) for name in input_names
        },
        "lscpu": command("lscpu"),
        "nvidia_smi": command(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ),
    }
    expected = {
        SPEC["historical_tmol"]["default_version"]: SPEC["historical_tmol"][
            "default_commit"
        ],
        SPEC["historical_tmol"]["nucleic_version"]: SPEC["historical_tmol"][
            "nucleic_commit"
        ],
        SPEC["latest_tmol"]["version"]: SPEC["latest_tmol"]["commit"],
    }
    errors = []
    for version, revision in expected.items():
        state = payload["tmol_sources"][version]
        if state["revision"] != revision:
            errors.append(
                f"tmol {version}: expected {revision}, found {state['revision']}"
            )
        if not state["clean"]:
            errors.append(f"tmol {version}: source worktree is dirty")
    if payload["harness"]["revision"] is None:
        errors.append("benchmark harness Git revision could not be resolved")
    elif not payload["harness"]["clean"]:
        errors.append("benchmark harness worktree is dirty")
    payload["validation_errors"] = errors
    payload["valid"] = not errors
    output = ROOT / "metadata/run_provenance.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(output)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
