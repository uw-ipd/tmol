#!/usr/bin/env python3
"""Validate that a tmol release contains the intended wheel matrix."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

WHEEL_RE = re.compile(
    r"^tmol-(?P<version>[^+]+)\+"
    r"(?P<local>cpu|cu\d+torch\d+\.\d+)-"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--cpu-count", type=int, required=True)
    parser.add_argument("--platform", action="append", default=[])
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="LOCAL:CP_TAG:ARCH",
    )
    args = parser.parse_args()

    paths = sorted(args.wheel_dir.glob("*.whl"))
    if len(paths) != len({path.name for path in paths}):
        raise SystemExit("duplicate wheel filenames found")

    try:
        wheels = [parse_wheel(path) for path in paths]
    except ValueError as error:
        raise SystemExit(str(error)) from error

    versions = {wheel.version for wheel in wheels}
    if len(versions) != 1:
        raise SystemExit(f"expected one release version, found: {sorted(versions)}")

    gpu = [wheel for wheel in wheels if wheel.local != "cpu"]
    cpu = [wheel for wheel in wheels if wheel.local == "cpu"]
    if len(gpu) != args.gpu_count:
        raise SystemExit(f"expected {args.gpu_count} GPU wheels, found {len(gpu)}")
    if len(cpu) != args.cpu_count:
        raise SystemExit(f"expected {args.cpu_count} CPU wheels, found {len(cpu)}")

    allowed_platforms = set(args.platform)
    found_platforms = {wheel.platform for wheel in wheels}
    unexpected = found_platforms - allowed_platforms
    if unexpected:
        raise SystemExit(f"unexpected platform tags: {sorted(unexpected)}")

    wheel_keys = {wheel.key for wheel in wheels}
    missing = set(args.require) - wheel_keys
    if missing:
        raise SystemExit(f"missing required wheel variants: {sorted(missing)}")

    print(
        f"Validated {len(wheels)} wheels for tmol {versions.pop()}: "
        f"{len(gpu)} GPU, {len(cpu)} CPU"
    )


if __name__ == "__main__":
    main()
