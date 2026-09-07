"""Capture the exact source and compiled artifacts used by a candidate A/B run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def git_state(path: Path) -> dict:
    status = command("git", "-C", str(path), "status", "--porcelain")
    # Ninja creates this empty transient lock in the source worktree while an
    # editable build is active. It is not an input and can briefly outlive the
    # build process, so do not mistake it for an unreproducible source change.
    if status is not None:
        status = "\n".join(
            line for line in status.splitlines() if line.strip() != "?? .ninja_lock"
        )
    return {
        "path": str(path),
        "revision": command("git", "-C", str(path), "rev-parse", "HEAD"),
        "clean": status == "",
        "status": status,
    }


def cmake_configuration(source: Path) -> dict[str, str]:
    wanted = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CXX_COMPILER",
        "TMOL_BUILD_TESTS",
    }
    values: dict[str, str] = {}
    for line in (source / "CMakeCache.txt").read_text().splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        name_and_type, value = line.split("=", 1)
        name = name_and_type.split(":", 1)[0]
        if name in wanted:
            values[name] = value
    return values


def build_state(source: Path) -> dict:
    binaries = sorted((source / "tmol").glob("_C*.so"))
    return {
        "source": git_state(source),
        "cmake": cmake_configuration(source),
        "extension_modules": [checked_file(path) for path in binaries],
    }


def freeze(environment: Path) -> list[str] | None:
    output = command(str(environment / "bin/python"), "-m", "pip", "freeze", "--all")
    return output.splitlines() if output is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-env", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--candidate-env", type=Path, required=True)
    parser.add_argument("--harness-root", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = build_state(args.baseline_source)
    candidate = build_state(args.candidate_source)
    payload = {
        "schema_version": 1,
        "captured_unix_time": time.time(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "platform": platform.platform(),
        "container": checked_file(args.image),
        "harness": git_state(args.harness_root),
        "baseline": baseline,
        "candidate": candidate,
        "python_environments": {
            "baseline": {
                "path": str(args.baseline_env),
                "pip_freeze": freeze(args.baseline_env),
            },
            "candidate": {
                "path": str(args.candidate_env),
                "pip_freeze": freeze(args.candidate_env),
            },
        },
        "lscpu": command("lscpu"),
        "nvidia_smi": command(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ),
    }
    errors: list[str] = []
    for label, state in (("baseline", baseline), ("candidate", candidate)):
        if not state["source"]["clean"]:
            errors.append(f"{label} source worktree is dirty")
        if not state["source"]["revision"]:
            errors.append(f"{label} source revision could not be resolved")
        if not state["extension_modules"]:
            errors.append(f"{label} compiled extension module is missing")
        if state["cmake"].get("CMAKE_BUILD_TYPE") != "Release":
            errors.append(f"{label} CMAKE_BUILD_TYPE is not Release")
        if state["cmake"].get("TMOL_BUILD_TESTS") != "OFF":
            errors.append(f"{label} TMOL_BUILD_TESTS is not OFF")
    for key in ("CMAKE_BUILD_TYPE", "CMAKE_CUDA_ARCHITECTURES", "TMOL_BUILD_TESTS"):
        if baseline["cmake"].get(key) != candidate["cmake"].get(key):
            errors.append(f"baseline and candidate differ in {key}")
    if not payload["harness"]["clean"]:
        errors.append("benchmark harness worktree is dirty")
    payload["validation_errors"] = errors
    payload["valid"] = not errors

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(args.output)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
