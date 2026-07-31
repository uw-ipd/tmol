"""
Score a protein
===============

Load a PDB file into a :class:`tmol.pose.pose_stack.PoseStack`, render the
beta2016 score function, evaluate the total weighted energy, and render the
scored protein in an interactive 3D viewer.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

import tmol
from tmol.io import pose_stack_from_pdb
from tmol.io.write_pose_stack_pdb import write_pose_stack_pdb
from tmol.score import beta2016_score_function


def pose_center(pose_stack):
    """Return the geometric center of the first pose for viewer labels."""
    coords = pose_stack.coords[0].detach().cpu()
    finite_coords = coords[torch.isfinite(coords).all(dim=-1)]
    center = finite_coords.mean(dim=0)
    return {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])}


def view_scored_pose(pose_stack, pdb_output_path, total_score):
    """Return a py3Dmol viewer with the weighted score shown as a label."""
    import py3Dmol

    write_pose_stack_pdb(pose_stack, str(pdb_output_path))

    view = py3Dmol.view(width=760, height=520)
    view.addModel(Path(pdb_output_path).read_text(), "pdb")
    view.setBackgroundColor("white")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"stick": {"radius": 0.12}})
    view.addLabel(
        f"beta2016 total score: {total_score:.3f}",
        {
            "position": pose_center(pose_stack),
            "backgroundColor": "white",
            "fontColor": "black",
            "fontSize": 14,
            "inFront": True,
            "showBackground": True,
        },
    )
    view.zoomTo()
    return view


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repo_root = Path(tmol.__file__).resolve().parents[1]
pdb_path = repo_root / "tmol" / "tests" / "data" / "pdb" / "1ubq.pdb"

pose_stack = pose_stack_from_pdb(str(pdb_path), device)
sfxn = beta2016_score_function(device)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)

score = scorer(pose_stack.coords)
protein_score = float(score[0])
print(f"1ubq beta2016 score: {protein_score:.3f}")

with TemporaryDirectory() as tmpdir:
    viewer = view_scored_pose(
        pose_stack,
        Path(tmpdir) / "1ubq_scored.pdb",
        protein_score,
    )

viewer
