import numpy
import torch

from tmol.io import pose_stack_from_pdb
from tmol.score.elec.elec_energy_term import ElecEnergyTerm
from tmol.pose.packed_block_types import PackedBlockTypes

from tmol.tests.score.common.test_energy_term import EnergyTermTestBase


def test_smoke(default_database, torch_device):
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)

    assert elec_energy.device == torch_device
    assert elec_energy.param_resolver.device == torch_device


def test_annotate_restypes(fresh_default_restype_set, default_database, torch_device):
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)

    rt_list = fresh_default_restype_set.residue_types
    pbt = PackedBlockTypes.from_restype_list(
        default_database.chemical, rt_list, torch_device
    )

    for rt in rt_list:
        elec_energy.setup_block_type(rt)
        assert hasattr(rt, "elec_partial_charge")
        assert hasattr(rt, "elec_inter_repr_path_distance")
        assert hasattr(rt, "elec_intra_repr_path_distance")
    elec_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "elec_partial_charge")
    assert hasattr(pbt, "elec_inter_repr_path_distance")
    assert hasattr(pbt, "elec_intra_repr_path_distance")

    assert pbt.elec_partial_charge.device == torch_device
    assert pbt.elec_inter_repr_path_distance.device == torch_device
    assert pbt.elec_intra_repr_path_distance.device == torch_device


def test_whole_pose_scoring_module_smoke(ubq_pdb, default_database, torch_device):
    gold_vals = numpy.array([[-0.428092]], dtype=numpy.float32)
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)
    r2_not_cterm = torch.zeros((1, 3, 2), dtype=torch.bool, device=torch_device)
    r2_not_cterm[0, 2, 1] = True
    p1 = pose_stack_from_pdb(
        ubq_pdb, torch_device, residue_end=3, res_not_connected=r2_not_cterm
    )
    for bt in p1.packed_block_types.active_block_types:
        elec_energy.setup_block_type(bt)
    elec_energy.setup_packed_block_types(p1.packed_block_types)
    elec_energy.setup_poses(p1)

    elec_pose_scorer = elec_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = elec_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)
    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


class TestElecEnergyTerm(EnergyTermTestBase):
    energy_term_class = ElecEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
        return super().test_whole_pose_scoring_10(
            ubq_pdb,
            default_database,
            torch_device,
            update_baseline=False,
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
        resnums = [(0, 4)]
        return super().test_whole_pose_scoring_gradcheck(
            ubq_pdb, default_database, torch_device, resnums=resnums
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
        resnums = [(0, 4)]
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
        resnums = [(0, 4)]
        return super().test_block_scoring_reweighted_gradcheck(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            nondet_tol=1e-6,  # fd this is necessary here...
        )
