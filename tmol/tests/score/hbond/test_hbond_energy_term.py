import numpy
import pytest
import torch

from tmol.io import pose_stack_from_pdb
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import IncludeCurrentSampler, build_rotamers
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


def test_paged_rotamer_scores_match_full_values_indices_and_gradients(
    ubq_pdb, default_database, torch_device, monkeypatch
):
    from tmol.score.hbond import _hbond_energy_term

    pose = pose_stack_from_pdb(ubq_pdb, torch_device, residue_end=10)
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    pose, rotamers = build_rotamers(
        pose,
        SetPackerTask.from_packer_task(task),
        pose.packed_block_types.chem_db,
    )
    term = HBondEnergyTerm(param_db=default_database, device=torch_device)
    for block in pose.packed_block_types.active_block_types:
        term.setup_block_type(block)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    scorer = term.render_rotamer_scoring_module(pose, rotamers)
    scorer.n_score_types = 1
    monkeypatch.setattr(_hbond_energy_term, "_PACK_HBOND_ROTAMER_CANDIDATE_WINDOW", 7)

    full_coords = rotamers.coords.detach().requires_grad_(True)
    full_scores, full_indices = scorer(full_coords)
    full_weights = torch.linspace(
        -1, 2, full_scores.numel(), dtype=full_scores.dtype, device=torch_device
    ).reshape_as(full_scores)
    (full_gradient,) = torch.autograd.grad(
        torch.sum(full_scores * full_weights), full_coords
    )

    paged_coords = rotamers.coords.detach().requires_grad_(True)
    pages = list(scorer.iter_packing_entries(paged_coords, topology_only=False))
    paged_scores = torch.cat([scores for scores, _ in pages], dim=1)
    paged_indices = torch.cat([indices for _, indices in pages], dim=1)
    (paged_gradient,) = torch.autograd.grad(
        torch.sum(paged_scores * full_weights), paged_coords
    )

    assert len(pages) > 1
    assert torch.equal(paged_indices, full_indices)
    torch.testing.assert_close(paged_scores, full_scores, rtol=0, atol=0)
    torch.testing.assert_close(paged_gradient, full_gradient, rtol=1e-5, atol=1e-5)

    topology_pages = list(
        scorer.iter_packing_entries(paged_coords.detach(), topology_only=True)
    )
    assert all(scores is None for scores, _ in topology_pages)
    assert torch.equal(
        torch.cat([indices for _, indices in topology_pages], dim=1), full_indices
    )

    from tmol.score._score_function import RotamerScoringModule

    weighted_scorer = RotamerScoringModule(
        torch.tensor([2.0], device=torch_device), [scorer]
    )
    weighted_pages = list(
        weighted_scorer._iter_weighted_sparse_entries(
            paged_coords.detach(), retain_shared_dispatch=False
        )
    )
    assert all(page_term is scorer for page_term, _, _ in weighted_pages)
    assert torch.equal(
        torch.cat([indices for _, indices, _ in weighted_pages], dim=1),
        full_indices,
    )
    torch.testing.assert_close(
        torch.cat([values for _, _, values in weighted_pages]),
        2 * full_scores.detach()[0],
        rtol=0,
        atol=0,
    )


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
    ubq_pdb, default_database, torch_device, dtype
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
        if gradient:
            (expected_grad,) = torch.autograd.grad(expected, coords, weights)
            (actual_grad,) = torch.autograd.grad(actual, coords, weights)
            torch.testing.assert_close(
                actual_grad, expected_grad, atol=tolerance, rtol=tolerance
            )
