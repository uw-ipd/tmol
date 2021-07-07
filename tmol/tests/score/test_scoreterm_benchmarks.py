import pytest


from tmol.utility.reactive import reactive_property

from tmol.score.score_components import ScoreComponentClasses, IntraScore

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.coords import coords_for
from tmol.score.modules.ljlk import LJScore, LKScore
from tmol.score.modules.lk_ball import LKBallScore
from tmol.score.modules.elec import ElecScore
from tmol.score.modules.cartbonded import CartBondedScore
from tmol.score.modules.dunbrack import DunbrackScore
from tmol.score.modules.hbond import HBondScore
from tmol.score.modules.rama import RamaScore
from tmol.score.modules.omega import OmegaScore

from tmol.system.score_support import score_method_to_even_weights_dict


def benchmark_score_pass(benchmark, score_system, benchmark_pass, coords):
    # Score once to prep graph
    total = score_system.intra_total(coords)

    if benchmark_pass == "full":

        @benchmark
        def run():
            total = score_system.intra_total(coords)
            total.backward()

            float(total)

            return total

    elif benchmark_pass == "forward":

        @benchmark
        def run():
            total = score_system.intra_total(coords)

            float(total)

            return total

    elif benchmark_pass == "backward":

        @benchmark
        def run():
            total.backward(retain_graph=True)
            return total

    else:
        raise NotImplementedError

    return run


@pytest.mark.parametrize(
    "score_system_weight_pair",
    [
        ({LJScore}, score_method_to_even_weights_dict(LJScore)),
        ({LKScore}, score_method_to_even_weights_dict(LKScore)),
        ({LKBallScore}, score_method_to_even_weights_dict(LKBallScore)),
        ({ElecScore}, score_method_to_even_weights_dict(ElecScore)),
        ({CartBondedScore}, score_method_to_even_weights_dict(CartBondedScore)),
        ({DunbrackScore}, score_method_to_even_weights_dict(DunbrackScore)),
        ({HBondScore}, score_method_to_even_weights_dict(HBondScore)),
        ({RamaScore}, score_method_to_even_weights_dict(RamaScore)),
        ({OmegaScore}, score_method_to_even_weights_dict(OmegaScore)),
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
