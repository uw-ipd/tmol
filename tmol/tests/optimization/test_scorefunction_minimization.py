import torch
import pytest

from tmol.pose.pose_stack import PoseStack
from tmol.pose.pose_stack_builder import PoseStackBuilder
from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.optimization.modules import DOFMaskingFunc
from tmol.score.score_function import ScoreFunction
from tmol.score.score_types import ScoreType


class CartesianSfxnNetwork(torch.nn.Module):
    def __init__(self, score_function, pose_stack, coord_mask=None):
        super(CartesianSfxnNetwork, self).__init__()

        self.whole_pose_scoring_module = (
            score_function.render_whole_pose_scoring_module(pose_stack)
        )
        self.coord_mask = coord_mask

        self.full_coords = pose_stack.coords
        if self.coord_mask is None:
            self.masked_coords = torch.nn.Parameter(pose_stack.coords)
        else:
            self.masked_coords = torch.nn.Parameter(pose_stack.coords[self.coord_mask])
        self.count = 0

    def forward(self):
        self.count += 1
        self.full_coords = DOFMaskingFunc.apply(
            self.masked_coords, self.coord_mask, self.full_coords
        )
        return self.whole_pose_scoring_module(self.full_coords)


class KinematicSfxnNetwork(torch.nn.Module):
    def __init__(self, score_function, pose_stack, dof_mask=None):
        super(KinematicSfxnNetwork, self).__init__()

        self.whole_pose_scoring_module = (
            score_function.render_whole_pose_scoring_module(pose_stack)
        )
        self.dof_mask = dof_mask
        # TO DO!!!


def test_minimize_w_pose_and_sfxn_smoke(rts_ubq_res, default_database, torch_device):
    pose_stack1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        rts_ubq_res[:4], torch_device
    )
    pose_stack5 = PoseStackBuilder.from_poses([pose_stack1] * 5, torch_device)

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_lj, 1.0)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    cart_sfxn_network = CartesianSfxnNetwork(sfxn, pose_stack5)
    optimizer = LBFGS_Armijo(cart_sfxn_network.parameters(), lr=0.1, max_iter=20)

    E0 = cart_sfxn_network.whole_pose_scoring_module(cart_sfxn_network.full_coords)
    print("E0", E0)

    def closure():
        optimizer.zero_grad()
        E = cart_sfxn_network()
        E.backward()
        return E

    optimizer.step(closure)

    E1 = cart_sfxn_network.whole_pose_scoring_module(cart_sfxn_network.full_coords)
    print("E1", E1)
    assert E1 < E0

    print("n sfxn evals:", cart_sfxn_network.count)


@pytest.mark.parametrize("n_poses", [1, 3, 10, 30])
@pytest.mark.benchmark(group=["minimize_pose_stack"])
def test_minimize_w_pose_and_sfxn_benchmark(
    benchmark, rts_ubq_res, default_database, torch_device, n_poses
):
    pose_stack1 = PoseStackBuilder.one_structure_from_polymeric_residues(
        rts_ubq_res, torch_device
    )
    pose_stack = PoseStackBuilder.from_poses([pose_stack1] * n_poses, torch_device)
    start_coords = pose_stack.coords.clone()

    sfxn = ScoreFunction(default_database, torch_device)
    sfxn.set_weight(ScoreType.fa_lj, 1.0)
    sfxn.set_weight(ScoreType.fa_lk, 0.8)

    @benchmark
    def run():
        pose_stack.coords[:] = start_coords

        cart_sfxn_network = CartesianSfxnNetwork(sfxn, pose_stack)
        optimizer = LBFGS_Armijo(cart_sfxn_network.parameters(), lr=0.1, max_iter=20)

        E0 = cart_sfxn_network.whole_pose_scoring_module(cart_sfxn_network.full_coords)

        def closure():
            optimizer.zero_grad()
            E = cart_sfxn_network()
            E.backward()
            return E

        optimizer.step(closure)

        E1 = cart_sfxn_network.whole_pose_scoring_module(cart_sfxn_network.full_coords)
        # print("E0", E0, "E1", E1)
        # assert E1 < E0

        # print("n sfxn evals:", cart_sfxn_network.count)

    run
