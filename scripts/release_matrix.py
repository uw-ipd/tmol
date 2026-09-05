#!/usr/bin/env python3
"""Define the Linux wheel matrix used by release and smoke workflows."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

PYTHON_VERSIONS = ("3.11", "3.12", "3.13", "3.14")
LINUX_ARCHES = ("x86_64", "aarch64")
CPU_TORCH_VERSIONS = ("2.13", "2.14")
RELEASE_PLATFORMS = {
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "macosx_14_0_arm64",
}


@dataclass(frozen=True)
class GpuFamily:
    """A PyTorch/CUDA combination expanded over Python and architecture."""

    torch: str
    cuda: str
    pythons: tuple[str, ...]
    arches: tuple[str, ...]
    container: str
    release_x86_runner: str = "ubuntu-22.04"
    cuda_archs: str = ""


GPU_FAMILIES = (
    GpuFamily(
        "2.8",
        "129",
        ("3.12",),
        LINUX_ARCHES,
        "nvcr.io/nvidia/pytorch:25.06-py3",
    ),
    GpuFamily(
        "2.9",
        "130",
        ("3.12",),
        LINUX_ARCHES,
        "nvidia/cuda:13.0.2-devel-ubuntu24.04",
    ),
    GpuFamily(
        "2.10",
        "130",
        ("3.12",),
        LINUX_ARCHES,
        "nvidia/cuda:13.0.2-devel-ubuntu24.04",
    ),
    GpuFamily(
        "2.11",
        "130",
        ("3.12",),
        LINUX_ARCHES,
        "nvidia/cuda:13.0.2-devel-ubuntu24.04",
    ),
    GpuFamily(
        "2.12",
        "132",
        PYTHON_VERSIONS,
        LINUX_ARCHES,
        "nvcr.io/nvidia/pytorch:26.04-py3",
        # Full-architecture builds in this family can exhaust hosted-runner RAM.
        release_x86_runner="self-hosted",
    ),
    GpuFamily(
        "2.13",
        "130",
        ("3.12", "3.13", "3.14"),
        LINUX_ARCHES,
        "nvidia/cuda:13.0.2-devel-ubuntu24.04",
    ),
    GpuFamily(
        "2.14",
        "132",
        PYTHON_VERSIONS,
        LINUX_ARCHES,
        "nvcr.io/nvidia/pytorch:26.04-py3",
    ),
    # Compatibility lanes outside each PyTorch release's primary CUDA family.
    GpuFamily(
        "2.12",
        "130",
        ("3.12",),
        ("x86_64",),
        "nvcr.io/nvidia/pytorch:26.04-py3",
    ),
    GpuFamily(
        "2.10",
        "128",
        ("3.12",),
        ("x86_64",),
        "nvcr.io/nvidia/pytorch:25.02-py3",
    ),
    # Colab needs explicit T4/A100/L4 images instead of the release default.
    GpuFamily(
        "2.11",
        "128",
        ("3.12", "3.13"),
        ("x86_64",),
        "nvcr.io/nvidia/pytorch:25.02-py3",
        cuda_archs="75;80;89",
    ),
)


def _torch_package_version(torch_version: str) -> str:
    return "2.9.1" if torch_version == "2.9" else f"{torch_version}.0"


def gpu_wheel_rows(*, release: bool = True) -> list[dict[str, object]]:
    """Return every CUDA lane, with release-specific runner routing."""
    rows: list[dict[str, object]] = []
    for family in GPU_FAMILIES:
        for python_version in family.pythons:
            for arch in family.arches:
                local_tag = f"cu{family.cuda}torch{family.torch}"
                row: dict[str, object] = {
                    "python-version": python_version,
                    "python-tag": python_version.replace(".", ""),
                    "torch-version": family.torch,
                    "torch-package-version": _torch_package_version(family.torch),
                    "torch-index-url": (
                        f"https://download.pytorch.org/whl/cu{family.cuda}"
                    ),
                    "local-tag": local_tag,
                    "enable-cuda": True,
                    "container-image": family.container,
                    "runs-on": (
                        (family.release_x86_runner if release else "ubuntu-22.04")
                        if arch == "x86_64"
                        else "ubuntu-24.04-arm"
                    ),
                    "arch": arch,
                    "label": (
                        f"py{python_version} pt{family.torch} "
                        f"cu{family.cuda} {arch}"
                    ),
                }
                if family.cuda_archs:
                    row["cuda-archs"] = family.cuda_archs
                    row["label"] += " Colab"
                if (
                    python_version == "3.12"
                    and family.torch == "2.8"
                    and arch == "x86_64"
                ):
                    row["gpu-runtime"] = True
                if python_version == "3.12" and family.torch in {"2.13", "2.14"}:
                    row["portability"] = True
                rows.append(row)
    return rows


def cpu_wheel_rows() -> list[dict[str, object]]:
    """Return every published manylinux CPU wheel lane."""
    rows: list[dict[str, object]] = []
    for torch_version in CPU_TORCH_VERSIONS:
        for python_version in PYTHON_VERSIONS:
            for arch in LINUX_ARCHES:
                row: dict[str, object] = {
                    "python-version": python_version,
                    "python-tag": python_version.replace(".", ""),
                    "torch-version": torch_version,
                    "torch-package-version": _torch_package_version(torch_version),
                    "torch-index-url": "https://download.pytorch.org/whl/cpu",
                    "local-tag": f"cputorch{torch_version}",
                    "enable-cuda": False,
                    "container-image": "unused-for-manylinux-cpu",
                    "runs-on": (
                        "ubuntu-22.04" if arch == "x86_64" else "ubuntu-24.04-arm"
                    ),
                    "arch": arch,
                    "label": f"py{python_version} pt{torch_version} CPU {arch}",
                }
                if python_version == "3.12":
                    row["portability"] = True
                rows.append(row)
    return rows


def macos_wheel_rows() -> list[dict[str, object]]:
    """Return every published Apple Silicon CPU wheel lane."""
    return [
        {
            "python-version": python_version,
            "python-tag": python_version.replace(".", ""),
            "torch-version": torch_version,
            "torch-package-version": _torch_package_version(torch_version),
            "local-tag": f"cputorch{torch_version}",
            "arch": "arm64",
            "label": f"py{python_version} pt{torch_version} CPU macOS arm64",
        }
        for torch_version in CPU_TORCH_VERSIONS
        for python_version in PYTHON_VERSIONS
    ]


def linux_wheel_rows() -> list[dict[str, object]]:
    """Return the complete CUDA and CPU Linux wheel matrix."""
    return gpu_wheel_rows(release=False) + cpu_wheel_rows()


def expected_wheel_keys() -> set[str]:
    """Return the exact release manifest, including macOS CPU wheels."""
    keys = {
        f"{row['local-tag']}:cp{row['python-tag']}:{row['arch']}"
        for row in linux_wheel_rows()
    }
    keys.update(
        f"{row['local-tag']}:cp{row['python-tag']}:{row['arch']}"
        for row in macos_wheel_rows()
    )
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", choices=("gpu", "linux-cpu", "macos", "linux"))
    args = parser.parse_args()
    rows = {
        "gpu": gpu_wheel_rows,
        "linux-cpu": cpu_wheel_rows,
        "macos": macos_wheel_rows,
        "linux": linux_wheel_rows,
    }[args.matrix]()
    print(json.dumps({"include": rows}, separators=(",", ":")))


if __name__ == "__main__":
    main()
