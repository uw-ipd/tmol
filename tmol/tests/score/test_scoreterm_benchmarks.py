import pytest

from tmol.utility.reactive import reactive_property

from tmol.score.device import TorchDevice
from tmol.score.score_components import ScoreComponentClasses, IntraScore

from tmol.score.modules.ljlk import LJScore, LKScore
from tmol.score.modules.lk_ball import LKBallScore
from tmol.score.modules.elec import ElecScore
from tmol.score.modules.cartbonded import CartBondedScore
from tmol.score.modules.dunbrack import DunbrackScore
from tmol.score.modules.hbond import HBondScore
from tmol.score.modules.rama import RamaScore
from tmol.score.modules.omega import OmegaScore


def benchmark_score_pass(benchmark, score_system, benchmark_pass, coords):
    # Score once to prep graph
    total = score_system.intra_score().total

    if benchmark_pass == "full":

        @benchmark
        def run():
            score_graph.reset_coords()

            result = score_graph.intra_total()
            # total = ?#
            # total.backward() ?# TODO: what should go here?

            # float(total)

            # return total
            return

    elif benchmark_pass == "forward":

        @benchmark
        def run():
            total = score_system.intra_total(coords)

            return  # TODO: what should be returned?
            # intra_score() used to to return an object with a member total that itself had a .device member?

    elif benchmark_pass == "backward":

        @benchmark
        def run():
            # total.backward(retain_graph=True) TODO: what should go here?
            # return total
            return

    else:
        raise NotImplementedError

    return run


@pytest.mark.parametrize(
    "score_system_weight_pair",
    [
        ({LJScore}, {"lj": 1.0}),
        ({LKScore}, {"lk": 1.0}),
        ({LKBallScore}, {"lk_ball": 1.0}),
        ({ElecScore}, {"elec": 1.0}),
        ({CartBondedScore}, {"cartbonded": 1.0}),
        ({DunbrackScore}, {"dunbrack": 1.0}),
        ({HBondScore}, {"hbond": 1.0}),
        ({RamaScore}, {"rama": 1.0}),
        ({OmegaScore}, {"omega": 1.0}),
    ],
)
@pytest.mark.parametrize("benchmark_pass", ["full", "forward", "backward"])
@pytest.mark.benchmark(group="score_components")
def test_end_to_end_score_graph(
    benchmark, benchmark_pass, score_system_weight_pair, torch_device, ubq_system
):
    target_system = ubq_system
    score_system_dict = score_system_weight_pair[0]
    weight_dict = score_system_weight_pair[1]
    score_system = ScoreSystem.build_for(target_system, score_system_dict, weight_dict)
    coords = coords_for(target_system, score_system)

    run = benchmark_score_pass(benchmark, score_system, benchmark_pass, coords)

    assert run.device == torch_device
