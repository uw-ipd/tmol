import toolz

import numpy
import pandas

import tmol.database

from tmol.score.hbond import HBondScoreGraph
import tmol.system.residue.packed


def hbond_score_comparison(rosetta_baseline):
    test_system = (
        tmol.system.residue.packed.PackedResidueSystem()
        .from_residues(rosetta_baseline.tmol_residues)
    )  # yapf: disable
    hbond_graph = HBondScoreGraph(
        **tmol.score.system_graph_params(test_system, requires_grad=False)
    )

    # Extract list of hbonds from packed system into summary table
    # via atom metadata
    h_i = hbond_graph.hbond_pairs.pairs["h"]
    a_i = hbond_graph.hbond_pairs.pairs["a"]
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
                n: hbond_graph.hbond_pairs.pairs[n]
                for n in hbond_graph.hbond_pairs.pairs.dtype.names
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
        .sort_index(axis="columns")
    ) # yapf: disable

    err_msg = (f"Mismatched bb hbond identification:\n{score_comparison}\n\n")

    numpy.testing.assert_allclose(
        numpy.nan_to_num(score_comparison["score_rosetta"].values),
        numpy.nan_to_num(score_comparison["score_tmol"].values),
        rtol=2e-2,
        err_msg=err_msg,
    )
