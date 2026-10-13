"""Google Colab bootstrap shared by the interactive TMol tutorials."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

TUTORIAL_REF = "main"
RAW_BASE = f"https://raw.githubusercontent.com/RosettaCommons/tmol/{TUTORIAL_REF}"
# Advance only after the release wheels have been published and exercised.
# The source project's next version may not have downloadable wheels yet.
TMOL_RELEASE = "0.1.64"


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
                "--only-binary=:all:",
                "--constraint",
                str(constraint),
                *requirements,
            ]
        )


def setup_colab(fixtures: list[str]) -> None:
    """Install the matching TMol wheel and download tutorial fixtures."""
    import torch
    from packaging.version import Version

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This tutorial requires CUDA-enabled PyTorch and a Colab GPU runtime. "
            "Choose a GPU under 'Runtime > Change runtime type', reconnect, "
            "and run again."
        )
    cuda_major = (torch.version.cuda or "").split(".")[0]
    if Version(torch.__version__) < Version("2.11") or cuda_major not in {"12", "13"}:
        raise RuntimeError(
            f"TMol {TMOL_RELEASE} needs PyTorch 2.11+ with CUDA 12 or 13; "
            f"this runtime has PyTorch {torch.__version__}, CUDA {torch.version.cuda}."
        )

    runtime_packages = [
        "itables>=2.0",
        "py3Dmol>=2.4,<3",
    ]
    _pip_install(
        [f"tmol[cu{cuda_major}]=={TMOL_RELEASE}", *runtime_packages], torch.__version__
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

    print(
        f"Colab GPU environment ready: {torch.cuda.get_device_name(0)}; "
        f"PyTorch {torch.__version__}."
    )
