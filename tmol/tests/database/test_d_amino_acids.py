"""The generated D-amino-acid residue types.

A D residue is the mirror image of its L form. Reflection preserves bond
lengths and bond angles and negates every dihedral, so the two forms must agree
on everything unsigned and disagree in sign on everything chiral.
"""

import cattr
import numpy
import pytest

from tmol.chemical._restypes import RefinedResidueType
from tmol.database import ParameterDatabase
from tmol.support.chemical._add_d_amino_acids import D_NAME3, d_name


def residues(param_db):
    return {r.name: r for r in param_db.chemical.residues}


def base_pairs(param_db):
    """(L, D) base residue type pairs."""
    by_name = residues(param_db)
    return [
        (by_name[name], by_name[d_name(name)])
        for name, rt in sorted(by_name.items())
        if rt.name == rt.base_name
        and rt.properties.polymer.sidechain_chirality == "l"
        and d_name(name) in by_name
    ]


def ideal_coords(residue_type):
    refined = cattr.structure(cattr.unstructure(residue_type), RefinedResidueType)
    xyz = refined.compute_ideal_coords()
    return {ic.name: numpy.asarray(xyz[i]) for i, ic in enumerate(refined.icoors)}


def chiral_volume(coords):
    """Signed volume at CA; its sign is the residue's handedness."""
    n, ca, c, cb = (coords[a] for a in ("N", "CA", "C", "CB"))
    return float(numpy.dot(numpy.cross(n - ca, c - ca), cb - ca))


def test_every_chiral_l_residue_has_a_mirror() -> None:
    param_db = ParameterDatabase.get_default()
    by_name = residues(param_db)
    unmirrored = [
        rt.name
        for rt in by_name.values()
        if rt.name == rt.base_name
        and rt.properties.polymer.sidechain_chirality == "l"
        and d_name(rt.name) not in by_name
    ]
    assert unmirrored == []
    # glycine is achiral, so it has no D form
    assert "DGLY" not in by_name


def test_d_residues_use_the_pdb_three_letter_code() -> None:
    """Deposited structures carry these codes, so input matching depends on them."""
    param_db = ParameterDatabase.get_default()
    for l_rt, d_rt in base_pairs(param_db):
        assert d_rt.name3 == D_NAME3[l_rt.name3]
        assert d_rt.io_equiv_class == D_NAME3[l_rt.io_equiv_class]
        # the letter addresses the L base type; a D residue is written X[DALA]
        assert d_rt.one_letter_code is None


def test_d_residues_reference_their_l_form() -> None:
    param_db = ParameterDatabase.get_default()
    for l_rt, d_rt in base_pairs(param_db):
        assert d_rt.rama_reference == l_rt.name
        assert d_rt.dunbrack_reference == l_rt.base_name
        assert d_rt.reference_mirrored is True
        assert d_rt.properties.polymer.sidechain_chirality == "d"


def test_d_residues_have_inverted_chirality() -> None:
    param_db = ParameterDatabase.get_default()
    for l_rt, d_rt in base_pairs(param_db):
        left = chiral_volume(ideal_coords(l_rt))
        right = chiral_volume(ideal_coords(d_rt))
        assert left == pytest.approx(-right, abs=1e-9), l_rt.name


def test_d_residues_keep_the_l_bond_lengths() -> None:
    """Reflection is an isometry, so unsigned geometry must be untouched."""
    param_db = ParameterDatabase.get_default()
    for l_rt, d_rt in base_pairs(param_db):
        left, right = ideal_coords(l_rt), ideal_coords(d_rt)
        for a, b, *_ in l_rt.bonds:
            assert numpy.linalg.norm(left[a] - left[b]) == pytest.approx(
                numpy.linalg.norm(right[a] - right[b]), abs=1e-6
            ), f"{l_rt.name} {a}-{b}"


def test_d_residues_share_the_l_partial_charges() -> None:
    param_db = ParameterDatabase.get_default()

    def charges(res_name):
        return {
            p.atom: p.charge
            for p in param_db.scoring.elec.atom_charge_parameters
            if p.res == res_name
        }

    for l_rt, d_rt in base_pairs(param_db):
        assert charges(d_rt.name) == charges(l_rt.name), l_rt.name
        assert charges(l_rt.name), f"{l_rt.name} has no charges to compare"


def test_d_cartbonded_negates_only_the_impropers() -> None:
    """Impropers hold chirality; lengths, angles and torsions do not."""
    param_db = ParameterDatabase.get_default()
    cart = param_db.scoring.cartbonded.residue_params
    compared = 0
    for l_rt, d_rt in base_pairs(param_db):
        if l_rt.name not in cart:
            continue
        compared += 1
        left, right = cart[l_rt.name], cart[d_rt.name]
        assert left.length_parameters == right.length_parameters
        assert left.angle_parameters == right.angle_parameters
        assert left.torsion_parameters == right.torsion_parameters
        assert len(left.improper_parameters) == len(right.improper_parameters)
        for li, ri in zip(left.improper_parameters, right.improper_parameters):
            assert (li.atm1, li.atm2, li.atm3, li.atm4) == (
                ri.atm1,
                ri.atm2,
                ri.atm3,
                ri.atm4,
            )
            assert ri.phi1 == pytest.approx(-li.phi1)
            assert (ri.k1, ri.k2, ri.k3) == (li.k1, li.k2, li.k3)
    assert compared > 15


def test_d_residues_take_termini_variants() -> None:
    param_db = ParameterDatabase.get_default()
    by_base = {}
    for rt in param_db.chemical.residues:
        by_base.setdefault(rt.base_name, set()).add(rt.name)
    for name in ("DALA", "DPRO", "DHIS_POS"):
        assert {name, f"{name}:nterm", f"{name}:cterm"} <= by_base[name]


def test_l_base_name_resolves_d_forms_without_touching_dna() -> None:
    """The helper keys on chirality, not on a leading D in the name."""
    from tmol.chemical import l_base_name

    param_db = ParameterDatabase.get_default()
    by_name = residues(param_db)

    def refined(name):
        return cattr.structure(cattr.unstructure(by_name[name]), RefinedResidueType)

    for l_name, d_name_ in (("ALA", "DALA"), ("CYD", "DCYD"), ("HIS_POS", "DHIS_POS")):
        assert l_base_name(refined(d_name_)) == l_name
        assert l_base_name(refined(l_name)) == l_name
    # a variant reports its base type's L name
    assert l_base_name(refined("DALA:nterm")) == "ALA"
    # DNA residue types start with D but are not d-amino acids
    for name in ("DA", "DC", "DG", "DT"):
        assert l_base_name(refined(name)) == name


def test_every_polymer_class_has_default_termini() -> None:
    """The mapping is derived, so a new residue type is covered without an edit."""
    from tmol.io import default_canonical_ordering

    co = default_canonical_ordering()
    mapping = co.restypes_default_termini_mapping
    uncovered = [
        equiv
        for equiv in co.restype_io_equiv_classes
        if equiv not in mapping and equiv not in ("HOH", "VRT")
    ]
    assert uncovered == []
    assert mapping["DAL"] == ("nterm", "cterm")
    assert mapping["DA"] == ("na5prime", "na3prime")
