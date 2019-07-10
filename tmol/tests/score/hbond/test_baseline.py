import toolz

import pandas

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.hbond import HBondScoreGraph
from tmol.score.score_graph import score_graph

from tmol.system.packed import PackedResidueSystem

from tmol.utility.cuda.synchronize import synchronize_if_available


def hbond_score_comparison(rosetta_baseline):
    test_system = PackedResidueSystem.from_residues(rosetta_baseline.tmol_residues)

    @score_graph
    class HBGraph(CartesianAtomicCoordinateProvider, HBondScoreGraph):
        pass

    hbond_graph = HBGraph.build_for(test_system, requires_grad=False)

    # Extract list of hbonds from packed system into summary table
    # via atom metadata
    tmol_hbond_total = hbond_graph.intra_score().total_hbond
    synchronize_if_cuda_available()

    named_atom_index = pandas.DataFrame(test_system.atom_metadata).set_index(
        ["residue_index", "atom_name"]
    )["atom_index"]
    rosetta_hbonds = toolz.curried.reduce(pandas.merge)(
        (
            rosetta_baseline.hbonds,
            (
                named_atom_index.rename_axis(["a_res", "a_atom"])
                .to_frame("a")
                .reset_index()
            ),
            (
                named_atom_index.rename_axis(["h_res", "h_atom"])
                .to_frame("h")
                .reset_index()
            ),
        )
    ).set_index(["a", "h"])

    return tmol_hbond_total, rosetta_hbonds


def test_pyrosetta_hbond_comparison(ubq_rosetta_baseline):
    # score_comparison = hbond_score_comparison(ubq_rosetta_baseline)

    tmol_hbtot, rosetta_hbs = hbond_score_comparison(ubq_rosetta_baseline)

    thr14_intra_hbE = -0.0828936
    # rosetta3 doesn't report intraresidue hbonds
    assert abs(sum(rosetta_hbs["energy"]) + thr14_intra_hbE - tmol_hbtot) < 2e-5
