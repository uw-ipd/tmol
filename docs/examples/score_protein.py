"""
Score a protein
===============

Load a PDB file into a :class:`tmol.pose.pose_stack.PoseStack`, render the
beta2016 score function, and evaluate the total weighted energy.
"""

from pathlib import Path

import torch

from tmol.io import pose_stack_from_pdb
from tmol.score import beta2016_score_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repo_root = Path(__file__).resolve().parents[2]
pdb_path = repo_root / "tmol" / "tests" / "data" / "pdb" / "1ubq.pdb"

pose_stack = pose_stack_from_pdb(str(pdb_path), device)
sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)

score = scorer(pose_stack.coords)
print(f"1ubq beta2016 score: {float(score[0]):.3f}")
