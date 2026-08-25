"""Round-trip DNA-containing structures through the biotite reader/writer.

The round-trip tests read with no_optH=True; OptH repacks hydrogens and flips
HIS/ASN/GLN, so the input is only recoverable with it off.
"""

import numpy
import pytest
import torch

from tmol.io import (
    pose_stack_from_biotite,
    biotite_from_pose_stack,
    default_canonical_ordering,
)

DNA_NAME3S = ("DA", "DC", "DG", "DT")


def _atom_set(co, structure):
    """Identify atoms by canonical index so alias spellings compare equal."""
    out = set()
    for i in range(structure.array_length()):
        res_name = structure.res_name[i]
        at_inds = co.restypes_atom_index_mapping[res_name]
        atom_name = structure.atom_name[i]
        assert atom_name in at_inds, f"{res_name} atom {atom_name} unknown to tmol"
        out.add(
            (
                structure.chain_id[i],
                int(structure.res_id[i]),
                res_name,
                at_inds[atom_name],
            )
        )
    return out


def test_dna_five_prime_phosphate_is_not_required():
    """P is a DNA mainchain atom but the 5' variant removes it, so it must not
    be treated as required -- otherwise the biotite reader drops every 5'
    nucleotide."""
    co = default_canonical_ordering()
    for name3 in DNA_NAME3S:
        assert co.restypes_mainchain_atoms[name3][0] == "P"
        assert co.restypes_required_mainchain_atoms[name3] == (
            "O5'",
            "C5'",
            "C4'",
            "C3'",
            "O3'",
        )
    # protein is unaffected: no terminus patch removes N, CA or C
    for name3 in ("ALA", "GLY", "PRO"):
        assert co.restypes_required_mainchain_atoms[name3] == ("N", "CA", "C")


@pytest.mark.parametrize("fixture", ["biotite_dna", "biotite_protein_dna"])
def test_dna_biotite_round_trip(fixture, request, torch_device):
    structure = request.getfixturevalue(fixture)
    co = default_canonical_ordering()

    pose_stack = pose_stack_from_biotite(structure, torch_device, no_optH=True)
    out = biotite_from_pose_stack(pose_stack)

    before, after = _atom_set(co, structure), _atom_set(co, out)

    def _named(diff):
        return sorted(
            (ch, ri, rn, co.restypes_ordered_atom_names[rn][ai])
            for ch, ri, rn, ai in diff
        )

    assert (
        after == before
    ), f"dropped: {_named(before - after)}\ngained: {_named(after - before)}"


@pytest.mark.parametrize("fixture", ["biotite_dna", "biotite_protein_dna"])
def test_dna_biotite_coords_preserved(fixture, request, torch_device):
    structure = request.getfixturevalue(fixture)
    co = default_canonical_ordering()
    pose_stack = pose_stack_from_biotite(structure, torch_device, no_optH=True)
    out = biotite_from_pose_stack(pose_stack)

    def keyed(s):
        """Key by canonical index so alias spellings compare equal."""
        return {
            (
                s.chain_id[i],
                int(s.res_id[i]),
                co.restypes_atom_index_mapping[s.res_name[i]][s.atom_name[i]],
            ): s.coord[i]
            for i in range(s.array_length())
        }

    a, b = keyed(structure), keyed(out)
    shared = set(a) & set(b)
    assert len(shared) == len(a)
    for k in shared:
        numpy.testing.assert_allclose(a[k], b[k], atol=1e-3)


@pytest.mark.parametrize("fixture", ["biotite_dna", "biotite_protein_dna"])
def test_dna_biotite_scores_are_finite(fixture, request, torch_device):
    from tmol import beta2016_score_function

    structure = request.getfixturevalue(fixture)
    pose_stack = pose_stack_from_biotite(structure, torch_device)
    sfxn = beta2016_score_function(torch_device)
    scores = sfxn.render_whole_pose_scoring_module(pose_stack).unweighted_scores(
        pose_stack.coords
    )
    assert not torch.any(torch.isnan(scores))
    assert not torch.any(torch.isinf(scores))


def test_dna_biotite_keeps_all_nucleotides(biotite_dna, torch_device):
    """1BNA is 2 x 12 nt; none may be silently filtered out."""
    pose_stack = pose_stack_from_biotite(biotite_dna, torch_device, no_optH=True)
    pbt = pose_stack.packed_block_types
    base = [
        pbt.active_block_types[i].name.split(":")[0]
        for i in pose_stack.block_type_ind64[0].tolist()
        if i >= 0
    ]
    assert len(base) == 24
    assert all(n in DNA_NAME3S for n in base)
