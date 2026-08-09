"""Google Colab bootstrap shared by the interactive TMol tutorials."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from urllib.request import urlopen, urlretrieve

TUTORIAL_REF = "kdidi/sphinx-docs"
RAW_BASE = f"https://raw.githubusercontent.com/uw-ipd/tmol/{TUTORIAL_REF}"
RELEASE_WHEEL_TORCH_MINOR = "2.10"
RELEASE_WHEEL_CUDA = "12.8"
RELEASE_WHEEL_PYTHON = (3, 12)
TMOL_WHEEL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.46/"
    "tmol-0.1.46+cu128torch2.10-cp312-cp312-manylinux_2_28_x86_64.whl"
)
SOURCE_INSTALL_MARKER = Path(tempfile.gettempdir()) / "tmol-colab-source-install.txt"


def _project_dependencies_without_torch() -> list[str]:
    """Read branch dependencies while preserving Colab's active PyTorch."""
    pyproject_url = f"{RAW_BASE}/pyproject.toml"
    with urlopen(pyproject_url, timeout=30) as response:
        pyproject = tomllib.loads(response.read().decode("utf-8"))

    dependencies = pyproject.get("project", {}).get("dependencies")
    if not isinstance(dependencies, list) or not dependencies:
        raise RuntimeError(
            f"Could not read [project].dependencies from {pyproject_url}"
        )

    def requirement_name(requirement: str) -> str:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if match is None:
            raise RuntimeError(f"Could not parse project dependency: {requirement!r}")
        return re.sub(r"[-_.]+", "-", match.group(1)).lower()

    return [
        requirement
        for requirement in dependencies
        if requirement_name(requirement) != "torch"
    ]


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


def _source_install_is_ready(torch_version: str) -> bool:
    expected = f"{TUTORIAL_REF}\n{torch_version}\n"
    if (
        not SOURCE_INSTALL_MARKER.exists()
        or SOURCE_INSTALL_MARKER.read_text(encoding="utf-8") != expected
    ):
        return False
    try:
        from tmol._cpp_lib import _ensure_loaded

        _ensure_loaded()
    except Exception:
        return False
    return True


def setup_colab(
    fixtures: list[str],
    *,
    install_tutorial_source: bool = False,
) -> None:
    """Install TMol for the active Colab PyTorch and download tutorial fixtures."""
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
    release_wheel_matches = (
        torch_version.startswith(f"{RELEASE_WHEEL_TORCH_MINOR}.")
        and torch.version.cuda == RELEASE_WHEEL_CUDA
        and sys.version_info[:2] == RELEASE_WHEEL_PYTHON
    )
    use_source_install = install_tutorial_source or not release_wheel_matches
    if use_source_install and not install_tutorial_source:
        print(
            f"The published v0.1.46 TMol wheel targets Python "
            f"{'.'.join(map(str, RELEASE_WHEEL_PYTHON))}, PyTorch "
            f"{RELEASE_WHEEL_TORCH_MINOR}, and CUDA {RELEASE_WHEEL_CUDA}; this "
            f"runtime provides Python {sys.version_info.major}.{sys.version_info.minor}, "
            f"PyTorch {torch_version}, and CUDA {torch.version.cuda}. Building "
            "the tutorial branch "
            "against the installed PyTorch instead; this takes longer than the "
            "prebuilt-wheel path."
        )

    runtime_packages = [
        "atomworks>=2.2",
        "itables>=2.0",
        "openbabel-wheel==3.1.1.22",
        "py3Dmol>=2.4,<3",
    ]
    if use_source_install:
        # Build the exact tutorial branch when its APIs are required or when
        # Colab's PyTorch does not match the ABI-specific release wheel.
        _pip_install(
            [
                *_project_dependencies_without_torch(),
                *runtime_packages,
                "scikit-build-core>=0.10",
                "pybind11>=2.12",
                "ninja",
                "packaging>=24.2",
                "cmake>=3.18,<4",
            ],
            torch_version,
        )
        if _source_install_is_ready(torch_version):
            print("Compatible tutorial-branch TMol build is already installed.")
        else:
            build_env = os.environ.copy()
            build_env.update(
                {
                    # Colab commonly assigns T4 (75), A100 (80), or L4 (89).
                    "CMAKE_CUDA_ARCHITECTURES": "75;80;89",
                    "TMOL_DISABLE_WHEEL_FETCH": "1",
                }
            )
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--no-build-isolation",
                    "--force-reinstall",
                    "--no-deps",
                    f"git+https://github.com/uw-ipd/tmol.git@{TUTORIAL_REF}",
                ],
                env=build_env,
            )
            SOURCE_INSTALL_MARKER.write_text(
                f"{TUTORIAL_REF}\n{torch_version}\n", encoding="utf-8"
            )
    else:
        _pip_install(
            [
                TMOL_WHEEL,
                *runtime_packages,
            ],
            torch_version,
        )

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

    import tmol

    if not use_source_install:
        # The release wheel predates these pure-Python viewer helpers. Overlay
        # only that module while retaining the wheel's matching CUDA extensions.
        viewer_path = Path("_tmol_tutorial_visualize.py")
        urlretrieve(f"{RAW_BASE}/tmol/io/visualize.py", viewer_path)
        spec = importlib.util.spec_from_file_location(
            "_tmol_tutorial_visualize", viewer_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load TMol tutorial visualization helpers")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in ("view", "switchable_view", "selection_gallery"):
            setattr(tmol, name, getattr(module, name))

    print(
        f"Colab GPU environment ready: {torch.cuda.get_device_name(0)}; "
        f"PyTorch {torch.__version__}."
    )
