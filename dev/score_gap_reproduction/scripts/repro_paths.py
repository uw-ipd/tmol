"""Portable paths shared by the reproduction commands."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get("TMOL_SCORE_GAP_OUTPUT", ROOT / "results")).resolve()


def pyrosetta_path():
    # An installed PyRosetta package needs no override. For an unpacked licensed
    # distribution, point PYROSETTA_PATH at the directory containing pyrosetta/.
    return Path(os.environ.get("PYROSETTA_PATH", "."))
