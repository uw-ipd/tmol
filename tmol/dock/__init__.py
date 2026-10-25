"""Docking of a small molecule into a rigid or flexible-sidechain receptor.

Dock one ligand of a protein-ligand complex ``pose_stack`` (a single-pose
``PoseStack`` holding the receptor and the ligand in any placement)::

    from tmol.dock import GridBackend, GASchedule, run_docking

    backend = GridBackend.from_pose_stack(pose_stack, param_db)
    result = run_docking(pose_stack, backend, schedule=GASchedule(npool=100))
    result.best_pose_stack, result.best_energies

The ligand is the pose's only non-polymer, non-water block; with several, name
it with ``ligand="ATP"``, ``ligand=("A", 301)`` (chain, residue number) or a block
index. To dock many ligands, build one complex and one backend per ligand; to let
pocket sidechains move, pass ``flexible_residues=select_pocket_residues(pose_stack)``.
"""

from .ga import (
    DockingResult,
    GASchedule,
    RandomPlacement,
    run_docking,
    select_pocket_residues,
)
from .ligand_aligner import LigandAligner
from .scoring_backend import ContinuousBackend, GridBackend

__all__ = [
    "ContinuousBackend",
    "DockingResult",
    "GASchedule",
    "GridBackend",
    "LigandAligner",
    "RandomPlacement",
    "run_docking",
    "select_pocket_residues",
]
