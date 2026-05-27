"""Smoke tests for the GenBonded scoring term.

Two tests — one per ligand example — exercise:

  1. Database loading (GenBondedDatabase.from_file).
  2. Python-side setup: setup_block_type, setup_packed_block_types, setup_poses.
  3. ScoreFunction construction with only gen_torsions enabled.
  4. Full scoring forward pass
"""

import torch
from types import SimpleNamespace

from tmol.database import ParameterDatabase
from tmol.io.pose_stack_from_biotite import pose_stack_from_biotite
from tmol.score.genbonded.genbonded_energy_term import GenBondedEnergyTerm
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _genbonded_only_sfxn(param_db: ParameterDatabase, device: torch.device):
    """ScoreFunction with only gen_torsions enabled (weight=1.0)."""
    sfxn = ScoreFunction(param_db, device)
    sfxn.set_weight(ScoreType.gen_torsions, 1.0)
    return sfxn


def _build_pose_stack(atom_array, device):
    """Build a PoseStack with ligand preparation from a biotite AtomArray."""
    param_db = ParameterDatabase.get_fresh_default()
    pose_stack = pose_stack_from_biotite(
        atom_array,
        device,
        prepare_ligands=True,
        param_db=param_db,
    )
    return pose_stack, param_db


def _run_setup(pose_stack, param_db, device):
    """Run the full Python-side genbonded setup; return the term."""
    term = GenBondedEnergyTerm(param_db=param_db, device=device)
    pbt = pose_stack.packed_block_types

    for bt in pbt.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pbt)
    term.setup_poses(pose_stack)

    return term


def _assert_setup_attributes(pbt):
    """Verify the expected tensors were attached to PackedBlockTypes."""
    assert hasattr(
        pbt, "genbonded_intra_subgraphs"
    ), "setup_packed_block_types did not attach genbonded_intra_subgraphs"
    assert hasattr(
        pbt, "genbonded_intra_subgraph_offsets"
    ), "setup_packed_block_types did not attach genbonded_intra_subgraph_offsets"
    assert hasattr(
        pbt, "genbonded_intra_params"
    ), "setup_packed_block_types did not attach genbonded_intra_params"
    assert hasattr(
        pbt, "genbonded_connection_bond_types"
    ), "setup_packed_block_types did not attach genbonded_connection_bond_types"

    subgraphs = pbt.genbonded_intra_subgraphs
    params = pbt.genbonded_intra_params

    # Combined subgraph tensor: (N, 5) = [tag, a0, a1, a2, a3]
    assert (
        subgraphs.ndim == 2 and subgraphs.shape[1] == 5
    ), f"genbonded_intra_subgraphs shape {subgraphs.shape} should be (N, 5)"
    assert (
        params.ndim == 2 and params.shape[1] == 5
    ), f"genbonded_intra_params shape {params.shape} should be (N, 5)"
    assert (
        subgraphs.shape[0] == params.shape[0]
    ), "subgraphs and params row counts must match"

    offsets = pbt.genbonded_intra_subgraph_offsets
    assert offsets.ndim == 1
    assert offsets.shape[0] == len(
        pbt.active_block_types
    ), "one offset per block type expected"


# ---------------------------------------------------------------------------
# Test: I4B  (small drug-like ligand, ~10 heavy atoms, in lysozyme 184L)
# ---------------------------------------------------------------------------


def test_i4b_genbonded_setup(cif_184l_with_i4b, torch_device):
    """I4B: genbonded Python setup (no C++ kernel) completes without error."""
    pose_stack, param_db = _build_pose_stack(cif_184l_with_i4b, torch_device)
    _run_setup(pose_stack, param_db, torch_device)
    _assert_setup_attributes(pose_stack.packed_block_types)


def test_i4b_genbonded_smoke(cif_184l_with_i4b, torch_device):
    """I4B: score with gen_torsions only and verify result is finite."""
    pose_stack, param_db = _build_pose_stack(cif_184l_with_i4b, torch_device)

    sfxn = _genbonded_only_sfxn(param_db, torch_device)
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
    scores = scorer.unweighted_scores(pose_stack.coords)

    assert scores.shape[0] == 1, "expected one score per score type"
    assert not torch.any(torch.isnan(scores)), "NaN in genbonded scores (i4b)"
    assert not torch.any(torch.isinf(scores)), "Inf in genbonded scores (i4b)"


# ---------------------------------------------------------------------------
# Test: HEM  (large macrocyclic ligand, ~43 heavy atoms, in cytochrome c 155C)
# ---------------------------------------------------------------------------


def test_hem_genbonded_setup(cif_155c_with_hem, torch_device):
    """HEM: genbonded Python setup (no C++ kernel) completes without error."""
    pose_stack, param_db = _build_pose_stack(cif_155c_with_hem, torch_device)
    _run_setup(pose_stack, param_db, torch_device)
    _assert_setup_attributes(pose_stack.packed_block_types)


def test_hem_genbonded_smoke(cif_155c_with_hem, torch_device):
    """HEM: score with gen_torsions only and verify result is finite.

    HEM contains Fe which is dropped during preparation (unsupported element).
    The remaining atoms still exercise the genbonded torsion lookup.
    """
    pose_stack, param_db = _build_pose_stack(cif_155c_with_hem, torch_device)

    sfxn = _genbonded_only_sfxn(param_db, torch_device)
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
    scores = scorer.unweighted_scores(pose_stack.coords)

    assert scores.shape[0] == 1, "expected one score per score type"
    assert not torch.any(torch.isnan(scores)), "NaN in genbonded scores (hem)"
    assert not torch.any(torch.isinf(scores)), "Inf in genbonded scores (hem)"


def test_genbonded_rotamer_noop_module(torch_device):
    param_db = ParameterDatabase.get_fresh_default()
    term = GenBondedEnergyTerm(param_db=param_db, device=torch_device)

    fake_rotamer_set = SimpleNamespace(
        n_rots_for_pose=torch.tensor([2], dtype=torch.int32, device=torch_device),
        coord_offset_for_rot=torch.tensor(
            [0, 10], dtype=torch.int32, device=torch_device
        ),
    )
    module = term.render_rotamer_scoring_module(None, fake_rotamer_set)

    coords = torch.zeros((20, 3), dtype=torch.float32, device=torch_device)
    sparse_scores = module(coords)

    assert sparse_scores.shape[0] == 1
    assert sparse_scores.shape[1:] == (1, 2, 2)
