#!/usr/bin/env python3
"""Validate that a tmol release contains the intended wheel matrix."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from release_matrix import RELEASE_PLATFORMS, expected_wheel_keys

WHEEL_RE = re.compile(
    r"^tmol-(?P<version>[^+]+)\+"
    r"(?P<local>cputorch\d+\.\d+|cu\d+torch\d+\.\d+)-"
    r"cp(?P<python>\d+)-cp(?P=python)-"
    r"(?P<platform>[^.]+)\.whl$"
)


@dataclass(frozen=True)
class Wheel:
    path: Path
    version: str
    local: str
    python: str
    platform: str

    @property
    def arch(self) -> str:
        if self.platform.endswith("_arm64"):
            return "arm64"
        if self.platform.endswith("_x86_64"):
            return "x86_64"
        if self.platform.endswith("_aarch64"):
            return "aarch64"
        return ""

    @property
    def key(self) -> str:
        return f"{self.local}:cp{self.python}:{self.arch}"


def parse_wheel(path: Path) -> Wheel:
    match = WHEEL_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unexpected wheel filename: {path.name}")
    wheel = Wheel(path=path, **match.groupdict())
    if not wheel.arch:
        raise ValueError(f"unsupported wheel architecture: {path.name}")
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_dir", type=Path)
    args = parser.parse_args()

    paths = sorted(args.wheel_dir.glob("*.whl"))
    try:
        wheels = [parse_wheel(path) for path in paths]
    except ValueError as error:
        raise SystemExit(str(error)) from error

    versions = {wheel.version for wheel in wheels}
    if len(versions) != 1:
        raise SystemExit(f"expected one release version, found: {sorted(versions)}")

    gpu = [wheel for wheel in wheels if not wheel.local.startswith("cpu")]
    cpu = [wheel for wheel in wheels if wheel.local.startswith("cpu")]
    expected_keys = expected_wheel_keys()
    expected_gpu_count = sum(not key.startswith("cpu") for key in expected_keys)
    expected_cpu_count = len(expected_keys) - expected_gpu_count
    if len(gpu) != expected_gpu_count:
        raise SystemExit(f"expected {expected_gpu_count} GPU wheels, found {len(gpu)}")
    if len(cpu) != expected_cpu_count:
        raise SystemExit(f"expected {expected_cpu_count} CPU wheels, found {len(cpu)}")

    found_platforms = {wheel.platform for wheel in wheels}
    unexpected = found_platforms - RELEASE_PLATFORMS
    if unexpected:
        raise SystemExit(f"unexpected platform tags: {sorted(unexpected)}")

    wheel_keys = [wheel.key for wheel in wheels]
    if len(wheel_keys) != len(set(wheel_keys)):
        raise SystemExit("duplicate Python/local-version/architecture variants found")

    found_keys = set(wheel_keys)
    missing = expected_keys - found_keys
    extra = found_keys - expected_keys
    if missing or extra:
        raise SystemExit(
            f"release manifest mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    print(
        f"Validated {len(wheels)} wheels for tmol {versions.pop()}: "
        f"{len(gpu)} GPU, {len(cpu)} CPU"
    )


if __name__ == "__main__":
    main()
