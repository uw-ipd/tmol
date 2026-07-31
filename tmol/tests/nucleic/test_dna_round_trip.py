"""Smoke tests: read a DNA-containing PDB into a PoseStack and write it back out."""

import pytest

from tmol.io.canonical_ordering import (
    default_canonical_ordering,
    default_packed_block_types,
    canonical_form_from_pdb,
)
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form
from tmol.io.pdb_parsing import parse_pdb, to_pdb
from tmol.io.write_pose_stack_pdb import atom_records_from_pose_stack

DNA_NAME3S = ("DA", "DC", "DG", "DT")


def _pose_stack(pdb_lines, torch_device):
    co = default_canonical_ordering()
    pbt = default_packed_block_types(torch_device)
    canonical_form = canonical_form_from_pdb(co, pdb_lines, torch_device)
    return pose_stack_from_canonical_form(co, pbt, *canonical_form)


def _atom_set(co, pdb_lines):
    """Identify each atom record by canonical atom index, not by name.

    Rosetta writes the alias spellings (1HB, 1H); tmol writes its canonical ones
    (HB2, H1). Resolving both through the alias mapping compares atom identity
    rather than naming convention.
    """
    records = parse_pdb(pdb_lines)
    out = set()
    for _, r in records.iterrows():
        resn, atomn = r["resn"], r["atomn"].strip()
        at_inds = co.restypes_atom_index_mapping[resn]
        assert atomn in at_inds, f"{resn} atom {atomn} unknown to tmol"
        out.add((r["chain"], r["resi"], resn, at_inds[atomn]))
    return out


def test_dna_restypes_in_canonical_ordering():
    co = default_canonical_ordering()
    for name3 in DNA_NAME3S:
        assert name3 in co.restype_io_equiv_classes
        assert co.restypes_mainchain_atoms[name3] == (
            "P",
            "O5'",
            "C5'",
            "C4'",
            "C3'",
            "O3'",
        )
        assert name3 in co.restypes_default_termini_mapping


@pytest.mark.parametrize("fixture", ["dna_pdb", "protein_dna_pdb"])
def test_dna_pose_stack_round_trip(fixture, request, torch_device):
    pdb_lines = request.getfixturevalue(fixture)
    co = default_canonical_ordering()
    pose_stack = _pose_stack(pdb_lines, torch_device)

    out = to_pdb(atom_records_from_pose_stack(pose_stack))

    before, after = _atom_set(co, pdb_lines), _atom_set(co, out)

    def _named(diff):
        return sorted(
            (ch, ri, rn, co.restypes_ordered_atom_names[rn][ai])
            for ch, ri, rn, ai in diff
        )

    assert (
        after == before
    ), f"dropped: {_named(before - after)}\ngained: {_named(after - before)}"


@pytest.mark.parametrize("fixture", ["dna_pdb", "protein_dna_pdb"])
def test_dna_pose_stack_coords_are_all_resolved(fixture, request, torch_device):
    """Every atom of every block gets a coordinate; nothing is left as nan."""
    pdb_lines = request.getfixturevalue(fixture)
    pose_stack = _pose_stack(pdb_lines, torch_device)
    pbt = pose_stack.packed_block_types
    for b, bt_ind in enumerate(pose_stack.block_type_ind64[0].tolist()):
        if bt_ind < 0:
            continue
        offset = pose_stack.block_coord_offset64[0, b].item()
        n_ats = pbt.n_atoms[bt_ind].item()
        block = pose_stack.coords[0, offset : offset + n_ats]
        assert (
            not block.isnan().any()
        ), f"nan coords in block {b} ({pbt.active_block_types[bt_ind].name})"


def test_dna_termini_block_types(dna_pdb, torch_device):
    """1BNA has no 5' phosphate and a free 3' OH, so both termini variants apply."""
    pose_stack = _pose_stack(dna_pdb, torch_device)
    pbt = pose_stack.packed_block_types
    names = [
        pbt.active_block_types[i].name
        for i in pose_stack.block_type_ind[0].tolist()
        if i >= 0
    ]
    assert names[0].endswith(":dna5prime")
    assert names[-1].endswith(":dna3prime")
    assert all(n.split(":")[0] in DNA_NAME3S for n in names)


def test_protein_dna_chain_composition(protein_dna_pdb, torch_device):
    """1YSA: two DNA chains and two protein chains in one PoseStack."""
    pose_stack = _pose_stack(protein_dna_pdb, torch_device)
    pbt = pose_stack.packed_block_types
    base = [
        pbt.active_block_types[i].name.split(":")[0]
        for i in pose_stack.block_type_ind[0].tolist()
        if i >= 0
    ]
    # chains A and B are 20 nt each; chains C and D are the bZIP monomers
    assert sum(n in DNA_NAME3S for n in base) == 40
    assert sum(n not in DNA_NAME3S for n in base) > 0
