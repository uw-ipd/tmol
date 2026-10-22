import torch

from tmol.kinematics import FoldForest, MoveMap
from tmol.optimization import run_kin_min
from tmol.pack import PackerPalette
from tmol.relax import fast_relax
from tmol.score import ScoreType, beta_nov16_dens_score_function

from .conftest import EM_MODEL, model_pose


def _torsion_min(pose_stack, sfxn, *, fold_forest, move_map, verbose):
    return run_kin_min(pose_stack, sfxn, fold_forest, move_map, verbose=verbose)


def test_torsion_space_relax_into_density(em_map, torch_device):
    """A short torsion-space FastRelax of 9I8J residues into EMD-52727."""
    pose_stack = model_pose(
        EM_MODEL, torch_device, first=240, last=249, density_map=em_map
    )
    sfxn = beta_nov16_dens_score_function(torch_device)
    fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
    move_map = MoveMap.from_pose_stack(pose_stack)
    move_map.move_all_named_torsions = True
    move_map.move_all_jumps = True

    def total(ps):
        return sfxn.render_whole_pose_scoring_module(ps)(ps.coords).sum()

    before = total(pose_stack)
    relaxed = fast_relax(
        pose_stack,
        sfxn,
        PackerPalette(),
        move_map,
        fold_forest,
        num_repeats=1,
        min_fn=_torsion_min,
    )
    assert relaxed.density_map is em_map
    assert sfxn.get_weight(ScoreType.elec_dens_fast) == 35.0
    after = total(relaxed)
    assert torch.isfinite(after)
    assert after < before
