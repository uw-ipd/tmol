import numpy
import torch

from tmol.score.constraint.constraint_energy_term import ConstraintEnergyTerm

from tmol.tests.score.common.test_energy_term import EnergyTermTestBase

"""
def test_smoke(default_database, torch_device: torch.device):
    constraint_energy = ConstraintEnergyTerm(
        param_db=default_database, device=torch_device
    )

    assert constraint_energy.device == torch_device
    assert constraint_energy.global_params.a_mu.device == torch_device


def test_annotate_constraint_conns(
    rts_ubq_res, default_database, torch_device: torch.device
):
    constraint_energy = ConstraintEnergyTerm(
        param_db=default_database, device=torch_device
    )

    bt_list = residue_types_from_residues(rts_ubq_res)
    pbt = PackedBlockTypes.from_restype_list(bt_list, torch_device)

    for bt in bt_list:
        constraint_energy.setup_block_type(bt)
        assert hasattr(bt, "constraint_connections")
    constraint_energy.setup_packed_block_types(pbt)
    assert hasattr(pbt, "constraint_conns")
    constraint_conns = pbt.constraint_conns
    constraint_energy.setup_packed_block_types(pbt)

    assert pbt.constraint_conns.device == torch_device
    assert (
        pbt.constraint_conns is constraint_conns
    )  # Test to make sure the parameters remain the same instance


def test_whole_pose_scoring_module_gradcheck_whole_pose(
    rts_ubq_res, default_database, torch_device: torch.device
):
    constraint_energy = ConstraintEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        constraint_energy.setup_block_type(bt)
    constraint_energy.setup_packed_block_types(p1.packed_block_types)
    constraint_energy.setup_poses(p1)

    constraint_pose_scorer = constraint_energy.render_whole_pose_scoring_module(p1)

    def score(coords):
        scores = constraint_pose_scorer(coords)
        return torch.sum(scores)

    gradcheck(score, (p1.coords.requires_grad_(True),), eps=1e-3, atol=1e-2, rtol=5e-3)


def test_whole_pose_scoring_module_single(
    rts_ubq_res, default_database, torch_device: torch.device
):
    gold_vals = numpy.array([[-3.25716]], dtype=numpy.float32)
    constraint_energy = ConstraintEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    for bt in p1.packed_block_types.active_block_types:
        constraint_energy.setup_block_type(bt)
    constraint_energy.setup_packed_block_types(p1.packed_block_types)
    constraint_energy.setup_poses(p1)

    constraint_pose_scorer = constraint_energy.render_whole_pose_scoring_module(p1)

    coords = torch.nn.Parameter(p1.coords.clone())
    scores = constraint_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )


def test_whole_pose_scoring_module_10(
    rts_ubq_res, default_database, torch_device: torch.device
):
    n_poses = 10
    gold_vals = numpy.tile(numpy.array([[-3.25716]], dtype=numpy.float32), (n_poses))
    constraint_energy = ConstraintEnergyTerm(
        param_db=default_database, device=torch_device
    )
    p1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        res=rts_ubq_res, device=torch_device
    )
    pn = PoseStackBuilder.from_poses([p1] * n_poses, device=torch_device)

    for bt in pn.packed_block_types.active_block_types:
        constraint_energy.setup_block_type(bt)
    constraint_energy.setup_packed_block_types(pn.packed_block_types)

    constraint_energy.setup_poses(pn)

    constraint_pose_scorer = constraint_energy.render_whole_pose_scoring_module(pn)

    coords = torch.nn.Parameter(pn.coords.clone())
    scores = constraint_pose_scorer(coords)

    # make sure we're still good
    torch.arange(100, device=torch_device)

    numpy.testing.assert_allclose(
        gold_vals, scores.cpu().detach().numpy(), atol=1e-5, rtol=1e-5
    )"""


class TestConstraintEnergyTerm(EnergyTermTestBase):
    energy_term_class = ConstraintEnergyTerm

    @classmethod
    def test_whole_pose_scoring_10(cls, ubq_pdb, default_database, torch_device):
        resnums = [(0, 10)]
        return super().test_whole_pose_scoring_10(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            update_baseline=True,
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
            ubq_pdb, default_database, torch_device, resnums=resnums, eps=1e-3
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
        resnums = [(0, 5)]
        return super().test_block_scoring(
            ubq_pdb,
            default_database,
            torch_device,
            resnums=resnums,
            update_baseline=True,
        )

    @classmethod
    def test_block_scoring_reweighted_gradcheck(
        cls, ubq_pdb, default_database, torch_device
    ):
        resnums = [(0, 4)]
        return super().test_block_scoring_reweighted_gradcheck(
            ubq_pdb, default_database, torch_device, resnums
        )
