"""Google Colab bootstrap shared by the interactive TMol tutorials."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

TUTORIAL_REF = "kdidi/sphinx-docs"
RAW_BASE = f"https://raw.githubusercontent.com/uw-ipd/tmol/{TUTORIAL_REF}"
TMOL_WHEEL = (
    "https://github.com/uw-ipd/tmol/releases/download/v0.1.46/"
    "tmol-0.1.46+cu128torch2.10-cp312-cp312-manylinux_2_28_x86_64.whl"
)


def setup_colab(
    fixtures: list[str],
    *,
    install_tutorial_source: bool = False,
) -> None:
    """Install the GPU wheel and download checked-in tutorial fixtures."""
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
    if not torch.__version__.startswith("2.10."):
        raise RuntimeError(
            f"The TMol Colab environment requires PyTorch 2.10, "
            f"found {torch.__version__}."
        )

    runtime_packages = [
        "atomworks>=2.2",
        "itables>=2.0",
        "openbabel-wheel==3.1.1.22",
        "py3Dmol>=2.4,<3",
    ]
    if install_tutorial_source:
        # v0.1.46 predates the merged DNA/RNA implementation. Build the exact
        # tutorial branch against Colab's installed PyTorch until a newer
        # release wheel is available.
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                *runtime_packages,
                "scikit-build-core>=0.10",
                "pybind11>=2.12",
                "ninja",
                "packaging>=24.2",
                "cmake>=3.18,<4",
            ]
        )
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
    else:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                TMOL_WHEEL,
                *runtime_packages,
            ]
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

    if not install_tutorial_source:
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
