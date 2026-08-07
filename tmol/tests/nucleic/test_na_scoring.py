"""Score DNA-containing poses with beta2016.

DNA cart_bonded params come from Rosetta's assembled set (IdealParametersDatabase):
intra-residue lengths/angles, base-planarity torsions, the O3'-P phosphodiester
geometry, and the HO3'/HO5' terminal geometry. Still uncovered: cart_impropers (the
3 CA-centred backbone impropers have no DNA analogue), gen_torsions (keyed on the
generic-potential ligand types), rama/omega/dunbrack (amino-acid only) and ref (no
DNA weights).
"""

import pytest
import torch

from tmol import beta2016_score_function
from tmol.score.score_types import ScoreType
from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form

# terms that must be non-zero on a nucleic-acid-only pose
NA_ACTIVE_TERMS = (
    ScoreType.fa_ljatr,
    ScoreType.fa_ljrep,
    ScoreType.fa_lk,
    ScoreType.fa_elec,
    ScoreType.hbond,
    ScoreType.lk_ball,
    ScoreType.lk_ball_iso,
    ScoreType.cart_lengths,
    ScoreType.cart_angles,
)

# base-planarity restraints; the kernel splits torsions and impropers so assert
# only their combined contribution
NA_PLANARITY_TERMS = (ScoreType.cart_torsions, ScoreType.cart_impropers)

# terms with no nucleic acid parameterization
NA_INACTIVE_TERMS = (
    ScoreType.gen_torsions,
    ScoreType.rama,
    ScoreType.omega,
    ScoreType.ref,
    ScoreType.dunbrack_rot,
    ScoreType.dunbrack_rotdev,
    ScoreType.dunbrack_semirot,
)


def _pose_stack(pdb_lines, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, pdb_lines, torch_device)
    return pose_stack_from_canonical_form(co, pbt, *canonical_form)


def _unweighted(sfxn, pose_stack):
    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    term_scores = wpsm(pose_stack.coords, sum_terms=False, apply_weights=False)
    return {st: term_scores[i, :] for i, st in enumerate(sfxn.all_score_types())}


@pytest.mark.parametrize("fixture", ["dna_pdb", "rna_pdb", "protein_dna_pdb"])
def test_beta2016_scores_na_are_finite(fixture, request, torch_device):
    pdb_lines = request.getfixturevalue(fixture)
    pose_stack = _pose_stack(pdb_lines, torch_device)
    sfxn = beta2016_score_function(torch_device)

    scores = _unweighted(sfxn, pose_stack)
    for st, val in scores.items():
        assert not torch.isnan(val).any(), f"{st} is nan"
        assert not torch.isinf(val).any(), f"{st} is inf"

    total = sfxn.render_whole_pose_scoring_module(pose_stack)(pose_stack.coords)
    assert torch.isfinite(total).all()


def test_beta2016_parameterized_terms_see_dna(dna_pdb, torch_device):
    """On a DNA-only pose every term with DNA parameters must be non-zero."""
    pose_stack = _pose_stack(dna_pdb, torch_device)
    sfxn = beta2016_score_function(torch_device)
    scores = _unweighted(sfxn, pose_stack)

    for st in NA_ACTIVE_TERMS:
        assert scores[st].abs().sum() > 0, f"{st} is zero on a DNA-only pose"

    planarity = sum(scores[st].abs().sum() for st in NA_PLANARITY_TERMS)
    assert planarity > 0, "no base-planarity restraint applied to DNA"


def test_beta2016_unparameterized_terms_are_zero_for_dna(dna_pdb, torch_device):
    """Pins the remaining gaps: no DNA ref/rotamer/generic-torsion parameters."""
    pose_stack = _pose_stack(dna_pdb, torch_device)
    sfxn = beta2016_score_function(torch_device)
    scores = _unweighted(sfxn, pose_stack)

    for st in NA_INACTIVE_TERMS:
        assert scores[st].abs().sum() == 0, f"{st} unexpectedly non-zero for DNA"


def test_beta2016_protein_dna_is_more_than_parts(protein_dna_pdb, torch_device):
    """The complex must score differently from its isolated chains, i.e. the
    protein-DNA interface actually contributes."""
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    sfxn = beta2016_score_function(torch_device)

    def total(lines):
        cf = canonical_form_from_pdb(co, lines, torch_device)
        ps = pose_stack_from_canonical_form(co, pbt, *cf)
        return sfxn.render_whole_pose_scoring_module(ps)(ps.coords).sum().item()

    lines = protein_dna_pdb.split("\n")
    dna_only = "\n".join(
        line for line in lines if line[:4] != "ATOM" or line[21:22] in ("A", "B")
    )
    prot_only = "\n".join(
        line for line in lines if line[:4] != "ATOM" or line[21:22] in ("C", "D")
    )

    complexed = total(protein_dna_pdb)
    separate = total(dna_only) + total(prot_only)
    assert complexed != pytest.approx(separate, abs=1.0)
