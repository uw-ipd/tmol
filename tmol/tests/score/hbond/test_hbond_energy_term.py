import numpy
import pytest
import torch

from tmol.io import pose_stack_from_pdb
from tmol.pose import PoseStackBuilder
from tmol.score.hbond import HBondEnergyTerm
from tmol.score import (
    ScoreFunction,
    ScoreType,
)
from tmol.tests.score.common import EnergyTermTestBase


def test_smoke(default_database, torch_device):
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)

    assert hbond_energy.device == torch_device
    assert hbond_energy.hb_param_db.global_param_table.device == torch_device
    assert hbond_energy.hb_param_db.pair_param_table.device == torch_device
    assert hbond_energy.hb_param_db.pair_poly_table.device == torch_device


def test_hbond_in_sfxn(default_database, torch_device):
    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.hbond, 1.0)
    assert len(sfxn.all_terms()) == 1
    assert isinstance(sfxn.all_terms()[0], HBondEnergyTerm)


def test_annotate_restypes(
    fresh_default_packed_block_types, default_database, torch_device
):
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)

    pbt = fresh_default_packed_block_types
    for rt in pbt.active_block_types:
        hbond_energy.setup_block_type(rt)
        assert hasattr(rt, "hbbt_params")
    hbond_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "hbpbt_params")

    assert pbt.hbpbt_params.tile_n_donH.device == torch_device
    assert pbt.hbpbt_params.tile_n_acc.device == torch_device
    assert pbt.hbpbt_params.tile_donH_inds.device == torch_device
    assert pbt.hbpbt_params.tile_acc_inds.device == torch_device
    assert pbt.hbpbt_params.tile_donorH_type.device == torch_device
    assert pbt.hbpbt_params.tile_acceptor_type.device == torch_device
    assert pbt.hbpbt_params.tile_acceptor_hybridization.device == torch_device
    assert pbt.hbpbt_params.is_hydrogen.device == torch_device


def test_whole_pose_scoring_module_smoke(ubq_pdb, default_database, torch_device):
    gold_vals = numpy.array([[-55.6756]], dtype=numpy.float32)
    hbond_energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    p1 = pose_stack_from_pdb(ubq_pdb, torch_device)
    for bt in p1.packed_block_types.active_block_types:
        hbond_energy.setup_block_type(bt)
    hbond_energy.setup_packed_block_types(p1.packed_block_types)
    hbond_energy.setup_poses(p1)

    hbond_pose_scorer = hbond_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = hbond_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)
    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


class TestHBondEnergyTerm(EnergyTermTestBase):
    energy_term_class = HBondEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
        return super().test_whole_pose_scoring_10(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        ubq_pdb,
        default_database,
        torch_device: torch.device,
    ):
        return super().test_whole_pose_scoring_jagged(
            ubq_pdb, default_database, torch_device, update_baseline=False
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(cls, ubq_pdb, default_database, torch_device):
        resnums = [(6, 8), (10, 12)]
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
        )

    @classmethod
    def test_block_scoring_matches_whole_pose_scoring(
        cls, ubq_pdb, default_database, torch_device
    ):
        return super().test_block_scoring_matches_whole_pose_scoring(
            ubq_pdb, default_database, torch_device
        )

    @classmethod
    def test_block_scoring(cls, ubq_pdb, default_database, torch_device):
        resnums = [(6, 8), (10, 12)]
        return super().test_block_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            update_baseline=False,
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, ubq_pdb, default_database, torch_device
    ):
        resnums = [(6, 8), (10, 12)]
        return super().test_block_scoring_reweighted_gradcheck(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compact_specialization_preserves_subsets(
    ubq_pdb, default_database, torch_device, dtype, monkeypatch
):
    if torch_device.type != "cuda":
        pytest.skip("CUDA compact interaction specialization")
    full = pose_stack_from_pdb(ubq_pdb, torch_device)
    short = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=10)
    pose = PoseStackBuilder.from_poses([full, short] * 32, torch_device)
    energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    for bt in pose.packed_block_types.active_block_types:
        energy.setup_block_type(bt)
    energy.setup_packed_block_types(pose.packed_block_types)
    energy.setup_poses(pose)
    term = energy.render_whole_pose_scoring_module(pose)
    coords = pose.coords.to(dtype).detach()
    neighbors = term.build_compact_block_neighbors(coords, term.block_neighbor_cutoff)
    assert pose.block_type_ind.numel() >= 4096
    assert pose._hbond_allow_split_pairs
    assert neighbors.numel() - 1 >= 32768

    # Reversed custom subsets with identical entries and different spare
    # capacity exercise split and combined kernels without rebuilding the list.
    subset = neighbors[1 : int(neighbors[0]) + 1 : 4].flip(0).clone()
    neighbors[0] = subset.numel()
    neighbors[1 : subset.numel() + 1] = subset
    compact = neighbors[: subset.numel() + 1].clone()
    assert compact.numel() - 1 < 32768
    weights = torch.linspace(-1, 2, 64, device=torch_device, dtype=dtype)[None, :]
    tolerance = 1e-10 if dtype == torch.float64 else 1e-5
    for gradient in (False, True):
        coords.requires_grad_(gradient)
        expected = term(coords, compact)
        actual = term(coords, neighbors)
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        # Low-level callers may omit the new optional dispatch hint.
        import tmol.score.hbond.potentials as potentials

        native = potentials.hbond_pose_scores
        with monkeypatch.context() as patch:
            patch.setattr(
                potentials, "hbond_pose_scores", lambda *args: native(*args[:-1])
            )
            legacy = term(coords, neighbors)
        torch.testing.assert_close(legacy, expected, atol=tolerance, rtol=tolerance)
        if gradient:
            (expected_grad,) = torch.autograd.grad(expected, coords, weights)
            (actual_grad,) = torch.autograd.grad(actual, coords, weights)
            (legacy_grad,) = torch.autograd.grad(legacy, coords, weights)
            torch.testing.assert_close(
                legacy_grad, expected_grad, atol=tolerance, rtol=tolerance
            )
            torch.testing.assert_close(
                actual_grad, expected_grad, atol=tolerance, rtol=tolerance
            )


@pytest.mark.parametrize("short_residues, expected_split", [(1, False), (40, True)])
def test_specialization_uses_actual_residue_count(
    ubq_pdb, default_database, torch_device, short_residues, expected_split
):
    full = pose_stack_from_pdb(ubq_pdb, torch_device)
    short = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=short_residues)
    pose = PoseStackBuilder.from_poses([full] + [short] * 63, torch_device)
    energy = HBondEnergyTerm(param_db=default_database, device=torch_device)
    for bt in pose.packed_block_types.active_block_types:
        energy.setup_block_type(bt)
    energy.setup_packed_block_types(pose.packed_block_types)
    energy.setup_poses(pose)
    assert pose._hbond_allow_split_pairs == (
        expected_split and torch_device.type == "cuda"
    )
