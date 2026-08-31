"""Google Colab bootstrap shared by the interactive TMol tutorials."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

TUTORIAL_REF = "master"
RAW_BASE = f"https://raw.githubusercontent.com/uw-ipd/tmol/{TUTORIAL_REF}"
TMOL_RELEASE = "0.1.52"
RELEASE_WHEEL_TORCH_MINOR = "2.11"
RELEASE_WHEEL_CUDA = "12.8"
RELEASE_WHEEL_PYTHONS = frozenset({(3, 12), (3, 13)})


def _wheel_url(python_version: tuple[int, int]) -> str:
    """Return the released Colab wheel URL for a supported Python runtime."""
    if python_version not in RELEASE_WHEEL_PYTHONS:
        raise ValueError(f"Unsupported Colab Python version: {python_version}")
    python_tag = "cp" + "".join(map(str, python_version))
    cuda_tag = RELEASE_WHEEL_CUDA.replace(".", "")
    build_tag = f"cu{cuda_tag}torch{RELEASE_WHEEL_TORCH_MINOR}"
    wheel = (
        f"tmol-{TMOL_RELEASE}+{build_tag}-{python_tag}-{python_tag}-"
        "manylinux_2_28_x86_64.whl"
    )
    return f"https://github.com/uw-ipd/tmol/releases/download/v{TMOL_RELEASE}/{wheel}"


def _pip_install(requirements: list[str], torch_version: str) -> None:
    """Install packages without allowing pip to replace the active PyTorch."""
    with tempfile.TemporaryDirectory(prefix="tmol-colab-") as temp_dir:
        constraint = Path(temp_dir) / "constraints.txt"
        constraint.write_text(f"torch=={torch_version}\n", encoding="utf-8")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--constraint",
                str(constraint),
                *requirements,
            ]
        )


def setup_colab(fixtures: list[str]) -> None:
    """Install the matching TMol wheel and download tutorial fixtures."""
    gpu_probe = subprocess.run(
        ["nvidia-smi"],
        check=False,
        capture_output=True,
        text=True,
    )
    if gpu_probe.returncode != 0:
        raise RuntimeError(
            "This tutorial requires a Colab GPU runtime. Choose "
            "'Runtime > Change runtime type > T4 GPU', reconnect, and run again."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A GPU is visible to Colab, but PyTorch cannot use CUDA. "
            "Reconnect to the GPU runtime and rerun the setup cell."
        )
    torch_version = torch.__version__
    python_version = sys.version_info[:2]
    if (
        not torch_version.startswith(f"{RELEASE_WHEEL_TORCH_MINOR}.")
        or torch.version.cuda != RELEASE_WHEEL_CUDA
        or python_version not in RELEASE_WHEEL_PYTHONS
    ):
        supported_pythons = ", ".join(
            ".".join(map(str, version)) for version in sorted(RELEASE_WHEEL_PYTHONS)
        )
        raise RuntimeError(
            f"The TMol v{TMOL_RELEASE} Colab wheels support Python "
            f"{supported_pythons}, PyTorch "
            f"{RELEASE_WHEEL_TORCH_MINOR}.x, and CUDA {RELEASE_WHEEL_CUDA}; "
            f"this runtime provides Python "
            f"{sys.version_info.major}.{sys.version_info.minor}, PyTorch "
            f"{torch_version}, and CUDA {torch.version.cuda}. Choose a supported "
            "Colab GPU runtime or build TMol from source."
        )

    runtime_packages = [
        "itables>=2.0",
        "openbabel-wheel==3.1.1.22",
        "py3Dmol>=2.4,<3",
    ]
    _pip_install([_wheel_url(python_version), *runtime_packages], torch_version)

    for relative_path in fixtures:
        destination = Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or destination.stat().st_size == 0:
            fixture_url = f"{RAW_BASE}/{relative_path}"
            partial = destination.with_name(f"{destination.name}.part")
            try:
                urlretrieve(fixture_url, partial)
                if partial.stat().st_size == 0:
                    raise RuntimeError(
                        f"Downloaded an empty fixture from {fixture_url}"
                    )
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)

    print(
        f"Colab GPU environment ready: {torch.cuda.get_device_name(0)}; "
        f"PyTorch {torch.__version__}."
    )
