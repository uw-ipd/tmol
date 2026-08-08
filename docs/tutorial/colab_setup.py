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
            "py3Dmol>=2.4,<3",
        ]
    )

    for relative_path in fixtures:
        destination = Path(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            urlretrieve(f"{RAW_BASE}/{relative_path}", destination)

    # The released wheel supplies matching compiled CUDA extensions. Load the
    # tutorial branch's pure-Python viewer helpers until this docs PR ships in a
    # release, then expose the same convenience functions used in the notebooks.
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
        "Colab environment ready. Select a GPU runtime; "
        "the tutorial will use CUDA when available."
    )
