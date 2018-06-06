import toolz

import numpy
import pandas

from tmol.utility.reactive import reactive_attrs

from tmol.score.coordinates import CartesianAtomicCoordinateProvider
from tmol.score.hbond import HBondScoreGraph

from tmol.system.packed import PackedResidueSystem
from tmol.system.score import extract_graph_parameters


def hbond_score_comparison(rosetta_baseline):
    test_system = (
        PackedResidueSystem.from_residues(rosetta_baseline.tmol_residues)
    )

    @reactive_attrs
    class HBGraph(
            CartesianAtomicCoordinateProvider,
            HBondScoreGraph,
    ):
        pass

    hbond_graph = HBGraph(
        **extract_graph_parameters(
            HBGraph,
            test_system,
            requires_grad=False,
        )
    )

    # Extract list of hbonds from packed system into summary table
    # via atom metadata
    h_i = hbond_graph.hbond_pair_metadata["h"]
    a_i = hbond_graph.hbond_pair_metadata["a"]
    tmol_candidate_hbonds = pandas.DataFrame.from_dict(
        toolz.merge(
            {
                "h_res": test_system.atom_metadata["residue_index"][h_i],
                "h_atom": test_system.atom_metadata["atom_name"][h_i],
                "a_res": test_system.atom_metadata["residue_index"][a_i],
                "a_atom": test_system.atom_metadata["atom_name"][a_i],
                "score": numpy.nan_to_num(hbond_graph.hbond_scores),
            },
            {
                n: hbond_graph.hbond_pair_metadata[n]
                for n in hbond_graph.hbond_pair_metadata.dtype.names
            },
        )
    ).set_index(["a", "h"])
    tmol_hbonds = tmol_candidate_hbonds.query("score != 0")

    del h_i, a_i

    # Merge with named atom index to get atom indicies in packed system
    # hbonds columns: ["a_atom", "a_res", "h_atom", "h_res", "energy"]
    named_atom_index = (
        pandas.DataFrame(test_system.atom_metadata)
        .set_index(["residue_index", "atom_name"])["atom_index"]
    )
    rosetta_hbonds = toolz.curried.reduce(pandas.merge)((
        rosetta_baseline.hbonds,
        (
            named_atom_index.rename_axis(["a_res", "a_atom"])
            .to_frame("a").reset_index()
        ),
        (
            named_atom_index.rename_axis(["h_res", "h_atom"])
            .to_frame("h").reset_index()
        ),
    )).set_index(["a", "h"])

    return pandas.merge(
        rosetta_hbonds["energy"].to_frame("score_rosetta"),
        tmol_hbonds.rename(columns={"score": "score_tmol"}),
        left_index=True,
        right_index=True,
        how="outer",
    )


def test_pyrosetta_hbond_comparison(ubq_rosetta_baseline):
    score_comparison = hbond_score_comparison(ubq_rosetta_baseline)

    score_comparison["score_delta"] = score_comparison.eval(
        "abs(score_tmol - score_rosetta)"
    )
    score_comparison["score_rel_delta"] = score_comparison.eval(
        "abs((score_tmol - score_rosetta) / ((score_tmol + score_rosetta) / 2))"
    )
    score_comparison = (
        score_comparison
        .sort_values(by="score_rel_delta")
        .rename(columns={
            "h_res": "res_h",
            "h_atom": "atom_h",
            "a_res": "res_a",
            "a_atom": "atom_a",
        })
        .reset_index()
        .drop(columns=["a", "h", "b0", "b", "d"])
        .sort_index(axis="columns")
    ) # yapf: disable

    inter_hbonds = score_comparison.query("res_h != res_a")
    numpy.testing.assert_allclose(
        numpy.nan_to_num(inter_hbonds["score_rosetta"].values),
        numpy.nan_to_num(inter_hbonds["score_tmol"].values),
        rtol=1e-4,
        err_msg=f"Mismatched bb hbond identification:\n{inter_hbonds}\n\n",
    )

    intra_hbonds = score_comparison.query("res_h == res_a")
    assert len(intra_hbonds) > 0
    numpy.testing.assert_allclose(
        numpy.full_like(intra_hbonds["score_tmol"].values, numpy.nan),
        intra_hbonds["score_rosetta"].values,
        err_msg=f"Intra-res bb hbond identification:\n{intra_hbonds}\n\n",
    )
