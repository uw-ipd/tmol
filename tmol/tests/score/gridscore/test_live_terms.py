"""Live ligand-vs-fixed-receptor hbond / LK-ball against the existing tmol terms.

The oracle is the stock term evaluated through block-pair scoring of the whole
receptor-ligand pose; the live kernels must give the same ligand-receptor
interface energies and (autograd) ligand-coordinate gradients on every device.
"""

import attr
import pytest
import torch

from tmol.pose import PoseStackBuilder
from tmol.score import ScoreFunction, ScoreType
from tmol.score.gridscore import (
    build_live_hbond_data,
    build_live_lk_ball_data,
    score_ligand_hbond,
    score_ligand_lk_ball,
)
from tmol.tests.ligand.test_ligand_receptor_perterm_baseline import (
    _GOLDEN_PER_TERM,
    _TARGETS,
    _TERM_ORDER,
)
from tmol.tests.ligand.test_protein_ligand_ddg import LIGAND_RES_NAME
from tmol.tests.score.gridscore.test_gridscore import _load_target_pose

_TERMS = {
    "hbond": (build_live_hbond_data, score_ligand_hbond, (ScoreType.hbond,)),
    "lk_ball": (
        build_live_lk_ball_data,
        score_ligand_lk_ball,
        (
            ScoreType.lk_ball_iso,
            ScoreType.lk_ball,
            ScoreType.lk_bridge,
            ScoreType.lk_bridge_uncpl,
        ),
    ),
}


def _ligand_block(pose_stack) -> int:
    (block,) = [
        b
        for b, t in enumerate(pose_stack.block_type_ind64[0].tolist())
        if t >= 0
        and pose_stack.packed_block_types.active_block_types[t].name3 == LIGAND_RES_NAME
    ]
    return block


def _ligand_coords(pose_stack, ligand: int) -> torch.Tensor:
    offset = int(pose_stack.block_coord_offset64[0, ligand])
    n_atoms = pose_stack.block_type(0, ligand).n_atoms
    return pose_stack.coords[:, offset : offset + n_atoms]


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("term", list(_TERMS))
def test_live_term_matches_existing_term(term, target, torch_device):
    build, score_live, score_types = _TERMS[term]
    pose_stack, param_db = _load_target_pose(target)
    ligand = _ligand_block(pose_stack)
    offset = int(pose_stack.block_coord_offset64[0, ligand])
    n_atoms = pose_stack.block_type(0, ligand).n_atoms
    n_poses = 4

    data = build(pose_stack, ligand, param_db).to(torch_device, torch.float64)
    generator = torch.Generator().manual_seed(0)
    native = _ligand_coords(pose_stack, ligand)
    noise = 0.1 * torch.randn((n_poses, n_atoms, 3), generator=generator)
    ligand_coords = (native + noise).to(torch_device, torch.float64)
    ligand_coords.requires_grad_(True)

    live = score_live(data, ligand_coords).reshape(n_poses, -1)
    (live_grad,) = torch.autograd.grad(live.sum(), ligand_coords)

    sfxn = ScoreFunction(param_db, torch_device)
    for score_type in score_types:
        sfxn.set_weight(score_type, 1.0)
    stack = PoseStackBuilder.from_poses([pose_stack] * n_poses, torch_device)
    scorer = sfxn.render_block_pair_scoring_module(stack)
    coords = stack.coords.clone().to(torch.float64)
    coords[:, offset : offset + n_atoms] = ligand_coords
    pair = scorer(coords, sum_terms=False, apply_weights=False)
    # Ligand-receptor interface: the ligand's row and column minus its diagonal.
    interface = pair[:, :, ligand].sum(-1) + pair[:, :, :, ligand].sum(-1)
    interface = interface - 2 * pair[:, :, ligand, ligand]
    (ref_grad,) = torch.autograd.grad(interface.sum(), ligand_coords)

    golden = _GOLDEN_PER_TERM[target]["interface"]
    if any(golden[_TERM_ORDER.index(t.name)] for t in score_types):
        assert live.abs().max() > 0  # the perturbed pose still has contacts
    torch.testing.assert_close(live, interface.T, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(live_grad, ref_grad, rtol=1e-5, atol=1e-5)


def test_live_lk_ball_rejects_covalently_connected_ligand():
    pose_stack, param_db = _load_target_pose("fxa")
    ligand = _ligand_block(pose_stack)
    receptor = next(
        b
        for b in range(pose_stack.max_n_blocks)
        if b != ligand and int(pose_stack.block_type_ind64[0, b]) >= 0
    )
    connections = pose_stack.inter_residue_connections64.clone()
    connections[0, ligand, 0] = torch.tensor((receptor, 0))
    connected = attr.evolve(pose_stack, inter_residue_connections64=connections)
    with pytest.raises(ValueError, match="no covalent receptor connections"):
        build_live_lk_ball_data(connected, ligand, param_db)
