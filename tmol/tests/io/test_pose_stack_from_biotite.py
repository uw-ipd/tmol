import biotite.structure
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, set_structure
import pytest
import torch

from tmol.database import ParameterDatabase
from tmol.io.canonical_ordering import CanonicalOrdering
from tmol.io.pose_stack_from_biotite import (
    build_context_from_biotite,
    canonical_form_from_biotite,
    pose_stack_from_biotite,
    biotite_from_pose_stack,
)

_CI_CIF_CODES = [
    "1UBQ",  # small, clean protein
    "1R21",  # NMR ensemble (multi-model)
    "1BL8",  # missing sidechains
    "6H9V",  # OXT edge case
    "3N0I",  # OXT edge case, multi-chain
    "6C4C",  # non-standard hydrogen naming
]


def _fetch_cif(pdb_code, tmp_path):
    """Fetch a CIF from RCSB and return the biotite structure."""
    from biotite.database import rcsb

    path = rcsb.fetch(pdb_code, "cif", target_path=tmp_path)
    return biotite.structure.io.load_structure(
        path, extra_fields=["occupancy", "b_factor"]
    )


@pytest.mark.parametrize("pdb_code", _CI_CIF_CODES)
def test_load_score_roundtrip_cif(pdb_code, torch_device, tmp_path):
    """Fetch a CIF from RCSB, build PoseStack, score, and re-export."""
    if torch_device != torch.device("cpu"):
        pytest.skip("CIF roundtrip only runs on CPU")

    from tmol import beta2016_score_function

    bt_struct = _fetch_cif(pdb_code, tmp_path)
    if isinstance(bt_struct, biotite.structure.AtomArrayStack):
        bt_struct = bt_struct[0]

    pose_stack = pose_stack_from_biotite(bt_struct, torch_device)
    sfxn = beta2016_score_function(torch_device)
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
    scores = scorer.unweighted_scores(pose_stack.coords)

    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))

    bio = biotite_from_pose_stack(pose_stack)
    out_file = CIFFile()
    set_structure(out_file, bio)
    out_file.write(tmp_path / f"{pdb_code}_roundtrip.cif")


def test_canonical_form_from_biotite(biotite_1r21, torch_device):
    canonical_form_from_biotite(biotite_1r21, torch_device=torch_device)


def test_pose_stack_from_biotite_1ubq(biotite_1ubq, torch_device):
    pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)


def test_pose_stack_from_biotite_return_context(biotite_1ubq, torch_device):
    pose_stack, context = pose_stack_from_biotite(
        biotite_1ubq, torch_device=torch_device, return_context=True
    )
    assert pose_stack is not None
    assert context.canonical_form is not None
    assert context.canonical_ordering is not None
    assert context.packed_block_types is not None


def test_build_context_from_biotite(biotite_1ubq, torch_device):
    context = build_context_from_biotite(biotite_1ubq, torch_device=torch_device)
    assert context.canonical_form.coords.device == torch_device


def test_build_context_honors_param_db_without_ligands(biotite_1ubq, torch_device):
    param_db = ParameterDatabase.get_fresh_default()
    context = build_context_from_biotite(
        biotite_1ubq,
        torch_device=torch_device,
        prepare_ligands=False,
        param_db=param_db,
    )
    expected_co = CanonicalOrdering.from_chemdb(param_db.chemical)
    assert context.parameter_database is param_db
    assert context.canonical_ordering.restype_io_equiv_classes == (
        expected_co.restype_io_equiv_classes
    )


def test_pose_stack_from_biotite_4tlm_cif(biotite_4tlm, torch_device):
    pose_stack_from_biotite(biotite_4tlm, torch_device=torch_device)


def test_pose_stack_from_and_to_biotite_1ubq(biotite_1ubq, torch_device):
    pose_stack = pose_stack_from_biotite(biotite_1ubq, torch_device=torch_device)
    biotite_atom_array = biotite_from_pose_stack(pose_stack)

    file = PDBFile()
    file.set_structure(biotite_atom_array)
    file.write("test_out.pdb")


def test_pose_stack_from_and_to_biotite_multiple_poses(biotite_1r21, torch_device):
    pose_stack = pose_stack_from_biotite(biotite_1r21, torch_device=torch_device)
    biotite_atom_array = biotite_from_pose_stack(pose_stack)

    file = PDBFile()
    file.set_structure(biotite_atom_array)
    file.write("test_out.pdb")


def test_canonical_form_multpose_metadata_propagation(biotite_1r21, torch_device):
    cf = canonical_form_from_biotite(biotite_1r21, torch_device=torch_device)
    assert cf.atom_b_factor is not None
    assert cf.atom_occupancy is not None
    assert cf.atom_b_factor.shape[0] == biotite_1r21.stack_depth()
    assert cf.atom_occupancy.shape[0] == biotite_1r21.stack_depth()
    assert float(cf.atom_b_factor[1].sum()) > 0.0
    assert float(cf.atom_occupancy[1].sum()) > 0.0


def test_pose_stack_from_biotite_1ubq_slice(biotite_1ubq, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1ubq)
    bt = biotite_1ubq[0 : starts[30]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_n_term(biotite_1r21, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1r21)
    bt = biotite_1r21[0][0 : starts[3]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_his_d(biotite_1r21, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1r21)
    bt = biotite_1r21[0][starts[52] : starts[55]]
    pose_stack_from_biotite(bt, torch_device=torch_device)


def test_pose_stack_from_biotite_missing_sidechain(biotite_1bl8, torch_device):
    bt = biotite_1bl8
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)

    biotite_atom_array = biotite_from_pose_stack(pose_stack)
    file = PDBFile()
    file.set_structure(biotite_atom_array)


def test_pose_stack_from_biotite_missing_single_sidechain(biotite_1bl8, torch_device):
    starts = biotite.structure.get_residue_starts(biotite_1bl8)
    bt = biotite_1bl8[starts[0] : starts[6]]
    pose_stack = pose_stack_from_biotite(bt, torch_device=torch_device)

    biotite_atom_array = biotite_from_pose_stack(pose_stack)
    file = PDBFile()
    file.set_structure(biotite_atom_array)
