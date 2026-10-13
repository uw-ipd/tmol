import os

import pytest
import torch

from tmol.score import ScoreFunction, ScoreType

# Rosetta beta_nov16 weighted energy at table grid nodes:
# 0.5 * RamaPrePro::eval_rpp_rama_score + 0.61 * P_AA::P_AA_pp_energy
# (residue, next residue, phi, psi) -> energy
ROSETTA_RAMA = {
    ("ALA", "PRO", -60, -40): 3.0667,
    ("ALA", "PRO", -60, 140): 1.9601,
    ("ALA", "PRO", -140, 130): 3.7606,
    ("ALA", "PRO", 60, 40): 4.5246,
    ("ALA", "ALA", -60, -40): -0.4532,
    ("ALA", "ALA", -60, 140): -0.7805,
    ("ALA", "ALA", -140, 130): 0.7868,
    ("ALA", "ALA", 60, 40): 1.0350,
    ("GLY", "PRO", -60, -40): 5.0688,
    ("GLY", "PRO", -60, 140): 4.7291,
    ("GLY", "PRO", -140, 130): 7.2943,
    ("GLY", "PRO", 60, 40): 4.2070,
    ("GLY", "ALA", -60, -40): 1.3911,
    ("GLY", "ALA", -60, 140): 1.3025,
    ("GLY", "ALA", -140, 130): 4.0461,
    ("GLY", "ALA", 60, 40): -0.2152,
    ("TYR", "PRO", -60, -40): 3.7810,
    ("TYR", "PRO", -60, 140): 2.3449,
    ("TYR", "PRO", -140, 130): 2.7143,
    ("TYR", "PRO", 60, 40): 4.2182,
    ("TYR", "ALA", -60, -40): 0.2349,
    ("TYR", "ALA", -60, 140): -0.2129,
    ("TYR", "ALA", -140, 130): 0.2471,
    ("TYR", "ALA", 60, 40): 0.9181,
}


@pytest.mark.parametrize("sfxn_file", ["beta2016.sfxn", "beta_soft.sfxn"])
def test_weighted_rama_matches_rosetta(default_database, sfxn_file):
    """Weighted rama at grid nodes equals Rosetta's, before proline and elsewhere.

    Guards two bugs: support/scoring/_rewrite_rama_binary.py renormalized each
    prepro table, but Rosetta's prepro probabilities are normalized jointly over
    all amino acids, which shifted every *_prepro table by a constant; and
    beta2016.sfxn / beta_soft.sfxn weighted rama by 0.5 although the tables
    already carry the 0.5 rama_prepro and 0.61 p_aa_pp weights.
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "database",
        "score_functions",
        sfxn_file,
    )
    sfxn = ScoreFunction.from_sfxn_file(path, default_database, torch.device("cpu"))
    weight = float(sfxn.get_weight(ScoreType.rama))
    tables = {t.table_id: t.table for t in default_database.scoring.rama.rama_tables}

    for (res, next_res, phi, psi), expected in ROSETTA_RAMA.items():
        table = tables[res + ("_prepro" if next_res == "PRO" else "")]
        value = weight * float(table[(phi + 180) // 10, (psi + 180) // 10])
        assert value == pytest.approx(expected, abs=1e-3), (res, next_res, phi, psi)
