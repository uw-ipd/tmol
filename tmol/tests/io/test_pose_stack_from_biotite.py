import biotite.structure
from biotite.structure.io.pdbx import CIFFile, set_structure
import pytest
import torch

from tmol.io.pose_stack_from_biotite import (
    build_context_from_biotite,
    canonical_form_from_biotite,
    pose_stack_from_biotite,
    biotite_from_pose_stack,
)
from tmol.tests.data import load_cif

_CI_CIF_CODES = [
    "1UBQ",  # small, clean protein
    "1R21",  # NMR ensemble (multi-model)
    "1BL8",  # missing sidechains
    "6H9V",  # OXT edge case
    "3N0I",  # OXT edge case, multi-chain
    "6C4C",  # non-standard hydrogen naming
    "3AA0",  # Crash on export with mismatch of CO and PBT n-atoms
]


@pytest.mark.parametrize("pdb_code", _CI_CIF_CODES)
def test_load_score_roundtrip_cif(pdb_code, tmp_path):
    """Load a local CIF, build PoseStack, score, and re-export."""
    device = torch.device("cpu")

    from tmol import beta2016_score_function

    bt_struct = load_cif(pdb_code)
    if isinstance(bt_struct, biotite.structure.AtomArrayStack):
        bt_struct = bt_struct[0]

    pose_stack = pose_stack_from_biotite(bt_struct, device)
    sfxn = beta2016_score_function(device)
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
    scores = scorer.unweighted_scores(pose_stack.coords)

    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))

    bio = biotite_from_pose_stack(pose_stack)
    out_file = CIFFile()
    set_structure(out_file, bio)
    out_file.write(tmp_path / f"{pdb_code}_roundtrip.cif")


def test_build_context_from_biotite_smoke(biotite_1ubq, torch_device):
    context = build_context_from_biotite(biotite_1ubq, torch_device=torch_device)
    assert context.canonical_form.coords.device == torch_device


def test_canonical_form_from_biotite_smoke(biotite_1r21, torch_device):
    canonical_form_from_biotite(biotite_1r21, torch_device=torch_device)


def test_pose_stack_from_biotite_1ubq_smoke(biotite_1ubq, torch_device):
    pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)


def test_pose_stack_from_biotite_1ubq_err_smoke(biotite_1ubq_err, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1ubq_err)
    bt = biotite_1ubq_err[0][starts[-5] : starts[-1]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_1ubq_cif_smoke(biotite_1ubq_cif, torch_device):
    pose_stack_from_biotite(biotite_1ubq_cif, torch_device=torch_device)


def test_pose_stack_from_and_to_biotite_1ubq_smoke(biotite_1ubq, torch_device):
    pose_stack = pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)
    biotite_from_pose_stack(pose_stack)


def test_pose_stack_from_and_to_biotite_multiple_poses_smoke(
    biotite_1r21, torch_device
):
    pose_stack = pose_stack_from_biotite(biotite_1r21, torch_device=torch_device)
    biotite_from_pose_stack(pose_stack)


def test_canonical_form_multipose_metadata_propagation(biotite_1r21, torch_device):
    cf = canonical_form_from_biotite(biotite_1r21, torch_device=torch_device)
    assert cf.atom_b_factor is not None
    assert cf.atom_occupancy is not None
    assert cf.atom_b_factor.shape[0] == biotite_1r21.stack_depth()
    assert cf.atom_occupancy.shape[0] == biotite_1r21.stack_depth()
    assert float(cf.atom_b_factor[1].sum()) > 0.0
    assert float(cf.atom_occupancy[1].sum()) > 0.0


def test_pose_stack_from_biotite_1ubq_slice_smoke(biotite_1ubq, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1ubq)
    bt = biotite_1ubq[0 : starts[30]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_n_term_smoke(biotite_1r21, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1r21)
    bt = biotite_1r21[0][0 : starts[3]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_c_term_smoke(biotite_1r21, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1r21)
    bt = biotite_1r21[0][starts[-5] : starts[-1]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_his_d_smoke(biotite_1r21, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1r21)
    bt = biotite_1r21[0][starts[52] : starts[55]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_missing_sidechain_smoke(biotite_1bl8, torch_device):
    bt = biotite_1bl8
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)
    biotite_from_pose_stack(pose_stack)


def test_pose_stack_from_biotite_missing_single_sidechain_smoke(
    biotite_1bl8, torch_device
):
    starts = biotite.structure.get_residue_starts(biotite_1bl8)
    bt = biotite_1bl8[starts[0] : starts[6]]
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)
    biotite_from_pose_stack(pose_stack)
