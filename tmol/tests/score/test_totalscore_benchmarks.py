import pytest
import torch


from tmol.score import TotalScoreGraph

from tmol.score.score_graph import score_graph
from tmol.score.device import TorchDevice

from tmol.score.coordinates import KinematicAtomicCoordinateProvider

from tmol.optimization.lbfgs_armijo import LBFGS_Armijo
from tmol.optimization.modules import TorsionalEnergyNetwork


@score_graph
class TotalScore(KinematicAtomicCoordinateProvider, FastScoreGraph, TorchDevice):
    pass


@pytest.fixture
def default_component_weights(torch_device):
    return {
        "total_lj": torch.tensor(1.0, device=torch_device),  # _rep 0.55 !
        "total_lk": torch.tensor(1.0, device=torch_device),
        "total_elec": torch.tensor(1.0, device=torch_device),
        "total_lk_ball": torch.tensor(0.92, device=torch_device),
        "total_lk_ball_iso": torch.tensor(-0.38, device=torch_device),
        "total_lk_ball_bridge": torch.tensor(-0.33, device=torch_device),
        "total_lk_ball_bridge_uncpl": torch.tensor(-0.33, device=torch_device),
        "total_hbond": torch.tensor(1.0, device=torch_device),
        "total_rama": torch.tensor(1.0, device=torch_device),  # renormalized
        "total_dun": torch.tensor(1.0, device=torch_device),  # renormalized
        "total_omega": torch.tensor(0.48, device=torch_device),
        "total_cartbonded_length": torch.tensor(1.0, device=torch_device),
        "total_cartbonded_angle": torch.tensor(1.0, device=torch_device),
        "total_cartbonded_torsion": torch.tensor(1.0, device=torch_device),
        "total_cartbonded_improper": torch.tensor(1.0, device=torch_device),
        "total_cartbonded_hxltorsion": torch.tensor(1.0, device=torch_device),
        ## unimplemented
        "total_dun_rot": torch.tensor(0.76, device=torch_device),
        "total_dun_dev": torch.tensor(0.69, device=torch_device),
        "total_dun_semi": torch.tensor(0.78, device=torch_device),
        "total_ref": torch.tensor(1.0, device=torch_device),
        "total_dslf": torch.tensor(1.25, device=torch_device),
    }


@pytest.mark.benchmark(group="total_score_setup")
@pytest.mark.parametrize("system_size", [40, 75, 150, 300])  # , 600])
def test_setup(
    benchmark, systems_bysize, system_size, torch_device, default_component_weights
):
    @benchmark
    def setup():
        score_graph = TotalScore.build_for(
            systems_bysize[system_size],
            requires_grad=True,
            device=torch_device,
            component_weights=default_component_weights,
        )
        return score_graph.intra_score().total

    score = setup
    assert score == score


@pytest.mark.benchmark(group="total_score_onepass")
@pytest.mark.parametrize("system_size", [40, 75, 150, 300])  # , 600])
def test_full(
    benchmark, systems_bysize, system_size, torch_device, default_component_weights
):
    score_graph = TotalScore.build_for(
        systems_bysize[system_size],
        requires_grad=True,
        device=torch_device,
        component_weights=default_component_weights,
    )
    score_graph.intra_score().total

    @benchmark
    def forward_backward():
        score_graph.reset_coords()
        total = score_graph.intra_score().total
        total.backward()
        return total

    forward_backward


@pytest.mark.benchmark(group="total_score_onepass")
@pytest.mark.parametrize("system_size", [40, 75, 150, 300])  # , 600])
def test_minimize_10steps(
    benchmark, systems_bysize, system_size, torch_device, default_component_weights
):
    score_graph = TotalScore.build_for(
        systems_bysize[system_size],
        requires_grad=True,
        device=torch_device,
        component_weights=default_component_weights,
    )
    score_graph.intra_score().total

    # score
    model = TorsionalEnergyNetwork(score_graph)

    # set tol to 0 so we are guaranteed to hit the iteration limit
    optimizer = LBFGS_Armijo(model.parameters(), lr=1.0, max_iter=10, atol=0, rtol=0)

    def closure():
        optimizer.zero_grad()
        score_graph.reset_coords()  # this line is necessary!

        E = model()
        E.backward()
        return E

    @benchmark
    def min_10_steps():
        optimizer.step(closure)

    min_10_steps
