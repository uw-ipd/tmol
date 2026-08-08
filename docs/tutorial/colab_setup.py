"""Google Colab bootstrap shared by the interactive TMol tutorials."""

from __future__ import annotations

import importlib.util
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


def setup_colab(fixtures: list[str]) -> None:
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

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            TMOL_WHEEL,
            "atomworks>=2.2",
            "itables>=2.0",
            "openbabel-wheel==3.1.1.22",
            "py3Dmol>=2.4,<3",
        ]
    )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A GPU is visible to Colab, but PyTorch cannot use CUDA. "
            "Reconnect to the GPU runtime and rerun the setup cell."
        )
    if not torch.__version__.startswith("2.10."):
        raise RuntimeError(
            f"The TMol Colab wheel requires PyTorch 2.10, found {torch.__version__}."
        )

    for relative_path in fixtures:
        destination = Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            urlretrieve(f"{RAW_BASE}/{relative_path}", destination)

    # The released wheel supplies matching compiled CUDA extensions. Load this
    # tutorial branch's pure-Python viewer helpers, then expose the same
    # convenience functions used in the notebooks.
    import tmol

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
