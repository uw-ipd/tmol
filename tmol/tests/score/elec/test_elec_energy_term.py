import numpy
import torch

from tmol.score.elec.elec_energy_term import ElecEnergyTerm
from tmol.pose.packed_block_types import residue_types_from_residues, PackedBlockTypes
from tmol.pose.pose_stack_builder import PoseStackBuilder

from tmol.tests.score.common.test_energy_term import EnergyTermTestBase
from tmol.tests.autograd import gradcheck


def test_smoke(default_database, torch_device):
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)

    assert elec_energy.device == torch_device
    assert elec_energy.param_resolver.device == torch_device


def test_annotate_restypes(ubq_res, default_database, torch_device):
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)

    rt_list = residue_types_from_residues(ubq_res)
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


def test_whole_pose_scoring_module_smoke(rts_ubq_res, default_database, torch_device):
    gold_vals = numpy.array([[-0.428092]], dtype=numpy.float32)
    elec_energy = ElecEnergyTerm(param_db=default_database, device=torch_device)
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_ubq_res[0:3], device=torch_device
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
    def test_whole_pose_scoring_10(
        cls, rts_ubq_res, default_database, torch_device, update_baseline=False
    ):
        return super().test_whole_pose_scoring_10(
            rts_ubq_res,
            default_database,
            torch_device,
            update_baseline,
            atol=1e-5,
            rtol=1e-5,
        )

    @classmethod
    def test_whole_pose_scoring_jagged(
        cls,
        rts_ubq_res,
        default_database,
        torch_device: torch.device,
        update_baseline=False,
    ):
        return super().test_whole_pose_scoring_jagged(
            rts_ubq_res, default_database, torch_device, update_baseline
        )

    @classmethod
    def test_whole_pose_scoring_gradcheck(
        cls, rts_ubq_res, default_database, torch_device
    ):
        return super().test_whole_pose_scoring_gradcheck(
            rts_ubq_res[0:4],
            default_database,
            torch_device,
            eps=1e-3,
            atol=5e-3,
            rtol=5e-3,
        )

    @classmethod
    def test_block_scoring(
        cls, rts_ubq_res, default_database, torch_device, update_baseline=False
    ):
        return super().test_block_scoring(
            rts_ubq_res[0:4], default_database, torch_device, update_baseline, 1e-4
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, rts_ubq_res, default_database, torch_device
    ):
        return super().test_block_scoring_reweighted_gradcheck(
            rts_ubq_res[0:4],
            default_database,
            torch_device,
            eps=1e-3,
            atol=1e-3,
            nondet_tol=1e-6,  # fd this is necessary here...
        )
