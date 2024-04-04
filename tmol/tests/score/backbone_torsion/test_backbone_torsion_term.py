import numpy
import torch

from tmol.score.backbone_torsion.bb_torsion_energy_term import BackboneTorsionEnergyTerm
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.tests.autograd import gradcheck
from tmol.tests.score.common.test_energy_term import EnergyTermTestBase


def test_smoke(default_database, torch_device):
    backbone_torsion_energy = BackboneTorsionEnergyTerm(
        param_db=default_database, device=torch_device
    )

    assert backbone_torsion_energy.device == torch_device


def test_annotate_restypes(
    fresh_default_packed_block_types, default_database, torch_device
):
    backbone_torsion_energy = BackboneTorsionEnergyTerm(
        param_db=default_database, device=torch_device
    )

    pbt = fresh_default_packed_block_types

    first_params = {}
    for bt in pbt.active_block_types:
        backbone_torsion_energy.setup_block_type(bt)
        assert hasattr(bt, "backbone_torsion_params")
        first_params[bt.name] = bt.backbone_torsion_params

    for bt in pbt.active_block_types:
        # test that block-type annotation is not repeated;
        # original annotation is still present in the bt
        backbone_torsion_energy.setup_block_type(bt)
        assert first_params[bt.name] is bt.backbone_torsion_params

    backbone_torsion_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "backbone_torsion_params")

    init_pbt_backbone_torsion_params = pbt.backbone_torsion_params
    backbone_torsion_energy.setup_packed_block_types(pbt)

    # test that the initial packed-block-types annotation
    # has not been repeated; initial annotation is still
    # present in the pbt
    assert init_pbt_backbone_torsion_params is pbt.backbone_torsion_params

    assert pbt.backbone_torsion_params.bt_rama_table.device == torch_device
    assert pbt.backbone_torsion_params.bt_omega_table.device == torch_device
    assert pbt.backbone_torsion_params.bt_upper_conn_ind.device == torch_device
    assert pbt.backbone_torsion_params.bt_is_pro.device == torch_device
    assert pbt.backbone_torsion_params.bt_backbone_torsion_atoms.device == torch_device


def test_whole_pose_scoring_module_smoke(rts_ubq_res, default_database, torch_device):
    gold_vals = numpy.array([[-12.743369], [4.100153]], dtype=numpy.float32)  # 4.284
    backbone_torsion_energy = BackboneTorsionEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        default_database.chemical, res=rts_ubq_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        backbone_torsion_energy.setup_block_type(bt)
    backbone_torsion_energy.setup_packed_block_types(p1.packed_block_types)
    backbone_torsion_energy.setup_poses(p1)

    backbone_torsion_pose_scorer = (
        backbone_torsion_energy.render_whole_pose_scoring_module(p1)
    )

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = backbone_torsion_pose_scorer(coords)

    # make sure we're still good
    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-3, rtol=0
    )


class TestBackboneTorsionEnergyTerm(EnergyTermTestBase):
    energy_term_class = BackboneTorsionEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(
        cls, rts_ubq_res, default_database, torch_device, update_baseline=False
    ):
        return super().test_whole_pose_scoring_10(
            rts_ubq_res, default_database, torch_device, update_baseline
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
        rts_ubq_res = rts_ubq_res[6:12]
        return super().test_whole_pose_scoring_gradcheck(
            rts_ubq_res,
            default_database,
            torch_device,
        )

    @classmethod
    def test_block_scoring(
        cls, rts_ubq_res, default_database, torch_device, update_baseline=False
    ):
        return super().test_block_scoring(
            rts_ubq_res[0:4], default_database, torch_device, update_baseline
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, rts_ubq_res, default_database, torch_device
    ):
        return super().test_block_scoring_reweighted_gradcheck(
            rts_ubq_res[6:12],
            default_database,
            torch_device,
        )
