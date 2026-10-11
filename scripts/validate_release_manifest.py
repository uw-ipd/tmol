#!/usr/bin/env python3
"""Validate that a tmol release contains the intended wheel matrix."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from release_matrix import RELEASE_PLATFORMS, expected_wheel_keys

WHEEL_RE = re.compile(
    r"^(?P<package>tmol|tmol_ops_cu12|tmol_ops_cu13)-(?P<version>[^+]+)-"
    r"cp(?P<python>\d+)-cp(?P=python)-"
    r"(?P<platform>[^.]+)\.whl$"
)


@dataclass(frozen=True)
class Wheel:
    path: Path
    version: str
    package: str
    python: str
    platform: str

    @property
    def local(self) -> str:
        return "cpu" if self.package == "tmol" else self.package.rsplit("_", 1)[1]

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

    expected_keys = expected_wheel_keys()

    found_platforms = {wheel.platform for wheel in wheels}
    unexpected = found_platforms - RELEASE_PLATFORMS
    if unexpected:
        raise SystemExit(f"unexpected platform tags: {sorted(unexpected)}")

    wheel_keys = [wheel.key for wheel in wheels]
    if len(wheel_keys) != len(set(wheel_keys)):
        raise SystemExit("duplicate Python/backend/architecture variants found")

    found_keys = set(wheel_keys)
    missing = expected_keys - found_keys
    extra = found_keys - expected_keys
    if missing or extra:
        raise SystemExit(
            f"release manifest mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    cpu_count = sum(wheel.local == "cpu" for wheel in wheels)
    print(
        f"Validated {len(wheels)} wheels for tmol {versions.pop()}: "
        f"{len(wheels) - cpu_count} GPU, {cpu_count} CPU"
    )


if __name__ == "__main__":
    main()
