"""Per-term scores for one structure of each noncanonical backbone class.

Canonical structures cannot exercise this: genbonded scores nothing on them by
construction, and the terms that negotiate over a torsion only disagree where a
residue is partly or wholly ligand-typed. One fixture per backbone class keeps
the four ways a residue can reach the scoring path under a pinned number.

The baseline is per score type rather than a total, so a failure names the term
that moved instead of only reporting that something did.
"""

import os

import pytest
import yaml

from tmol.database import ParameterDatabase
from tmol.io import atom_array_from_cif, pose_stack_from_biotite
from tmol.ligand import prepare_ligands
from tmol.score import beta2016_score_function
from tmol.tests.data import data_path

# one fixture per way a noncanonical residue reaches the scoring path
FIXTURES = {
    "alpha_aa": "collagen_hyp_1bkv",  # HYP: Rosetta backbone, ligand sidechain
    "nonstandard_aa": "beta_peptide_3c3g",  # B3D/B3E/B3K/B3L/B3Q/BAL: all ligand
    "dna": "na_dna_5mc_1d17",  # 5CM: Rosetta backbone, ligand base
    "nonstandard_na": "na_dna_ttd_1ttd",  # TTD: all ligand
}

BASELINE = data_path("noncanonical_scores.yaml")

# set to regenerate; the values are only as correct as the code that wrote them,
#    so review a diff rather than refreshing on a whim
UPDATE_BASELINE = False


def _score_by_term(stem, torch_device):
    param_db = ParameterDatabase.get_default()
    structure = atom_array_from_cif(data_path("ncaa_fixtures") / (stem + ".cif"))
    prepared, _ordering = prepare_ligands(structure, param_db=param_db)
    pose_stack = pose_stack_from_biotite(structure, torch_device, param_db=prepared)

    sfxn = beta2016_score_function(torch_device, param_db=prepared)
    module = sfxn.render_whole_pose_scoring_module(pose_stack)
    values = module(pose_stack.coords, sum_terms=False, apply_weights=False)
    values = values.detach().cpu().numpy()
    return {
        st.name: round(float(values[i].sum()), 4)
        for i, st in enumerate(sfxn.all_score_types())
    }


@pytest.mark.parametrize("backbone_class", sorted(FIXTURES))
def test_noncanonical_scores_match_baseline(backbone_class, torch_device):
    stem = FIXTURES[backbone_class]
    scores = _score_by_term(stem, torch_device)

    if UPDATE_BASELINE:
        existing = {}
        if os.path.exists(BASELINE):
            with open(BASELINE) as infile:
                existing = yaml.safe_load(infile) or {}
        existing[backbone_class] = scores
        with open(BASELINE, "w") as outfile:
            yaml.safe_dump(existing, outfile, default_flow_style=False, sort_keys=True)

    assert os.path.exists(
        BASELINE
    ), "no baseline for noncanonical scoring; re-run with UPDATE_BASELINE = True"
    with open(BASELINE) as infile:
        gold = yaml.safe_load(infile)
    assert backbone_class in gold, backbone_class

    moved = {
        name: (gold[backbone_class].get(name), value)
        for name, value in scores.items()
        if gold[backbone_class].get(name) != pytest.approx(value, abs=1e-3, rel=1e-4)
    }
    assert not moved, "%s (%s): %s" % (backbone_class, stem, moved)


def test_every_noncanonical_class_scores_gen_torsions(torch_device):
    """A noncanonical residue has torsions no Rosetta term claims.

    The partition hands those to genbonded, so a zero here means the boundary
    has moved and something is now going unconstrained.
    """
    for backbone_class, stem in sorted(FIXTURES.items()):
        scores = _score_by_term(stem, torch_device)
        assert scores["gen_torsions"] != 0.0, "%s (%s)" % (backbone_class, stem)
