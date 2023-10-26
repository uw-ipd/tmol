import toolz

import pandas

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.coords import coords_for
from tmol.score.modules.hbond import HBondScore
from tmol.system.packed import PackedResidueSystem
from tmol.system.score_support import score_method_to_even_weights_dict


def hbond_score_comparison(rosetta_baseline):
    test_system = PackedResidueSystem.from_residues(rosetta_baseline.tmol_residues)

    hbond_system = ScoreSystem.build_for(
        test_system, {HBondScore}, score_method_to_even_weights_dict(HBondScore)
    )
    coords = coords_for(test_system, hbond_system)

    # Extract list of hbonds from packed system into summary table
    # via atom metadata
    tmol_hbond_total = hbond_system.intra_total(coords)

    named_atom_index = pandas.DataFrame(test_system.atom_metadata).set_index(
        ["residue_index", "atom_name"]
    )["atom_index"]

    return tmol_hbond_total, rosetta_baseline.hbonds


def test_pyrosetta_hbond_comparison(ubq_rosetta_baseline):
    # score_comparison = hbond_score_comparison(ubq_rosetta_baseline)

    tmol_hbtot, rosetta_hbs = hbond_score_comparison(ubq_rosetta_baseline)

    thr14_intra_hbE = -0.0828936
    # rosetta3 doesn't report intraresidue hbonds
    assert abs(sum(rosetta_hbs["energy"]) + thr14_intra_hbE - tmol_hbtot) < 2e-5
