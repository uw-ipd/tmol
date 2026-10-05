"""Tests for the wheel matrix and its release-manifest validator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from release_matrix import (
    cpu_wheel_rows,
    expected_wheel_keys,
    gpu_wheel_rows,
    linux_wheel_rows,
    macos_wheel_rows,
)


def test_release_matrix_is_complete_and_unique():
    release_gpu = gpu_wheel_rows()
    smoke_linux = linux_wheel_rows()

    assert len(release_gpu) == 34
    assert len(cpu_wheel_rows()) == 8
    assert len(macos_wheel_rows()) == 4
    assert len(smoke_linux) == len(release_gpu) + len(cpu_wheel_rows())
    assert len(expected_wheel_keys()) == 46
    assert len(
        {(row["python-tag"], row["local-tag"], row["arch"]) for row in smoke_linux}
    ) == len(smoke_linux)

    assert {
        (row["torch-version"], row["python-version"], row["arch"])
        for row in cpu_wheel_rows() + macos_wheel_rows()
    } == {
        ("2.14", python_version, arch)
        for python_version in ("3.11", "3.12", "3.13", "3.14")
        for arch in ("x86_64", "aarch64", "arm64")
    }


def test_matrix_cli_emits_workflow_json():
    script = Path(__file__).with_name("release_matrix.py")
    for name, expected_count in {
        "gpu": len(gpu_wheel_rows()),
        "linux-cpu": len(cpu_wheel_rows()),
        "macos": len(macos_wheel_rows()),
        "linux": len(linux_wheel_rows()),
    }.items():
        result = subprocess.run(
            [sys.executable, str(script), name],
            check=True,
            text=True,
            capture_output=True,
        )
        assert len(json.loads(result.stdout)["include"]) == expected_count


@pytest.mark.parametrize(
    "missing_wheel, error",
    [
        ("*cu132torch2.14-cp314-cp314-*.whl", "expected 34 GPU wheels, found 33"),
        (
            "*cputorch2.14-cp311-cp311-manylinux_2_28_x86_64.whl",
            "expected 12 CPU wheels, found 11",
        ),
        (
            "*cputorch2.14-cp314-cp314-macosx_14_0_arm64.whl",
            "expected 12 CPU wheels, found 11",
        ),
    ],
)
def test_manifest_validator_accepts_only_the_complete_matrix(
    tmp_path: Path, missing_wheel: str, error: str
):
    platform_for_arch = {
        "x86_64": "manylinux_2_28_x86_64",
        "aarch64": "manylinux_2_28_aarch64",
        "arm64": "macosx_14_0_arm64",
    }
    for key in expected_wheel_keys():
        local, python_tag, arch = key.split(":")
        (
            tmp_path
            / (
                f"tmol-9.9.9+{local}-{python_tag}-{python_tag}-"
                f"{platform_for_arch[arch]}.whl"
            )
        ).touch()

    command = [
        sys.executable,
        str(Path(__file__).with_name("validate_release_manifest.py")),
        str(tmp_path),
    ]
    valid = subprocess.run(command, text=True, capture_output=True)
    assert valid.returncode == 0, valid.stderr
    assert f"Validated {len(expected_wheel_keys())} wheels" in valid.stdout

    next(tmp_path.glob(missing_wheel)).unlink()
    incomplete = subprocess.run(command, text=True, capture_output=True)
    assert incomplete.returncode != 0
    assert error in incomplete.stderr
