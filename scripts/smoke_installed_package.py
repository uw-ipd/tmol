#!/usr/bin/env python3
"""Load and score with an installed wheel, outside the source checkout."""

import argparse
import sys
from importlib.metadata import version
from pathlib import Path

import torch
import tmol
from tmol._cpp_lib import _ensure_loaded, _find_extension_library
from tmol.io import pose_stack_from_pdb
from tmol.score import beta2016_score_function

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--version", required=True)
parser.add_argument("--pdb", type=Path, required=True)
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
args = parser.parse_args()
assert version("tmol") == args.version, version("tmol")
assert Path(tmol.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
_ensure_loaded()
assert _find_extension_library()
device = torch.device(args.device)
pose = pose_stack_from_pdb(str(args.pdb), device)
scorer = beta2016_score_function(device).render_whole_pose_scoring_module(pose)
coords = pose.coords.detach().clone().requires_grad_(True)
score = scorer(coords)
assert score.numel() == 1 and torch.isfinite(score).all(), score
score.sum().backward()
assert coords.grad is not None and torch.isfinite(coords.grad).all()
print(f"tmol {version('tmol')}, torch {torch.__version__}, {device}: {score.tolist()}")
