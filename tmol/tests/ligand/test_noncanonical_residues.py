"""Preparation of noncanonical polymer residues.

A noncanonical residue is prepared as a molecule -- its polymer connections are
replaced by chemical stubs, the ligand pipeline types and charges the capped
molecule, then the stubs are stripped back to connections. These tests cover
that path end to end from real structures, and the routing that sends a
polymer-linking residue down it instead of the free-molecule ligand path.

The assertions are sanity checks, not goldens: conformer generation is
stochastic, so what is pinned is the chemistry (atom set, connections, backbone
typing, torsions, integral net charge) and that the rebuilt geometry is
physically reasonable.
"""

from __future__ import annotations

import itertools

import biotite.structure as struc
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pytest

from tmol.database import ParameterDatabase
from tmol.ligand import (
    LigandPreparationError,
    chem_comp_types_from_cif,
    is_polymer_linking_ccd_type,
    prepare_ligands,
    prepare_polymer_residue,
)
from tmol.ligand._preparation import _ideal_coords_by_name
from tmol.ligand._polymer_profile import ALPHA_AA
from tmol.ligand._registry import rebuild_canonical_ordering
from tmol.tests.data import data_path

FIXTURE_DIR = data_path("ncaa_fixtures")

# A component id the CCD cannot define
UNDEFINED_CODE = "X_"

# stem -> (residue code, net formal charge at pH 7.4, expected chi count)
_FIXTURES: dict[str, tuple[str, float, int]] = {
    "phosphopeptide_5ema": ("SEP", -2.0, 3),
    "collagen_hyp_1bkv": ("HYP", 0.0, 1),
}


def _load(stem: str) -> struc.AtomArray:
    """Read a fixture with its bond table, as the polymer path requires."""
    cif = pdbx.CIFFile.read(str(FIXTURE_DIR / f"{stem}.cif"))
    return pdbx.get_structure(cif, model=1, include_bonds=True)


def _residue(atom_array: struc.AtomArray, res_name: str) -> struc.AtomArray:
    """The first instance of ``res_name`` in ``atom_array``."""
    matches = np.nonzero(atom_array.res_name == res_name)[0]
    assert len(matches) > 0, f"{res_name} not present in fixture"
    start = matches[0]
    mask = (
        (atom_array.res_name == res_name)
        & (atom_array.res_id == atom_array.res_id[start])
        & (atom_array.chain_id == atom_array.chain_id[start])
    )
    return atom_array[mask]


def _prepare(stem: str, res_name: str):
    param_db = ParameterDatabase.get_default()
    residue = _residue(_load(stem), res_name)
    prep = prepare_polymer_residue(
        residue, rebuild_canonical_ordering(param_db), param_db
    )
    return residue, prep


def _assert_alpha_amino_acid(residue: struc.AtomArray, prep) -> None:
    """Every invariant an alpha-amino-acid residue type must satisfy."""
    restype = prep.residue_type
    names = {a.name for a in restype.atoms}

    assert len(names) == len(restype.atoms), "duplicate atom names"

    # the caps are scaffolding; none of them may survive into the residue type
    assert names.isdisjoint(ALPHA_AA.cap_names)

    # every heavy atom of the input residue is carried through
    input_heavy = {
        str(n)
        for n, e in zip(residue.atom_name, residue.element)
        if str(e).strip().upper() != "H"
    }
    assert input_heavy <= names

    connections = {c.name: c.atom for c in restype.connections}
    assert connections == {"down": "N", "up": "C"}

    polymer = restype.properties.polymer
    assert polymer.polymer_type == "amino_acid"
    assert polymer.backbone_type == "alpha"
    assert polymer.sidechain_chirality == "l"
    assert tuple(polymer.mainchain_atoms) == ALPHA_AA.mainchain_atoms

    types = {a.name: a.atom_type for a in restype.atoms}
    assert types["CA"] == "CAbb"
    assert types["C"] == "CObb"
    assert types["O"] == "OCbb"
    assert types["N"] in ALPHA_AA.amide_n_types

    torsions = [t.name for t in restype.torsions]
    assert torsions[:3] == ["phi", "psi", "omega"]
    chis = torsions[3:]
    # chis are numbered from the backbone outwards with no gaps
    assert chis == [f"chi{i + 1}" for i in range(len(chis))]

    # every atom is reachable in the icoor tree
    assert {ic.name for ic in restype.icoors} >= names

    charges = prep.partial_charges
    assert set(charges) == names
    # mol2 partial charges are stored to four decimals, so the residue's
    #    formal charge is only recovered to about a millicharge
    net = sum(charges.values())
    assert net == pytest.approx(round(net), abs=1e-3)

    coords = _ideal_coords_by_name(restype)
    for a, b, *_rest in restype.bonds:
        d = float(np.linalg.norm(coords[a] - coords[b]))
        assert 0.9 < d < 2.0, f"{a}-{b} bond length {d:.3f} A is not chemical"
    # and no two atoms have collapsed onto each other
    for a, b in itertools.combinations(sorted(names), 2):
        d = float(np.linalg.norm(coords[a] - coords[b]))
        assert d > 0.7, f"{a} and {b} are {d:.3f} A apart"


@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_prepare_polymer_residue(stem: str) -> None:
    res_name, net_charge, n_chi = _FIXTURES[stem]
    residue, prep = _prepare(stem, res_name)

    assert prep.residue_type.name == res_name
    _assert_alpha_amino_acid(residue, prep)
    assert sum(prep.partial_charges.values()) == pytest.approx(net_charge, abs=1e-3)
    assert len([t for t in prep.residue_type.torsions if t.name.startswith("chi")]) == (
        n_chi
    )


def test_hydroxyproline_nitrogen_is_substituted() -> None:
    """HYP closes its sidechain onto N, which therefore carries no hydrogen."""
    _residue_array, prep = _prepare("collagen_hyp_1bkv", "HYP")
    restype = prep.residue_type
    types = {a.name: a.atom_type for a in restype.atoms}
    assert types["N"] == "Npro"
    bonded_to_n = {b for a, b, *_ in restype.bonds if a == "N"} | {
        a for a, b, *_ in restype.bonds if b == "N"
    }
    assert not any(types[n].startswith("H") for n in bonded_to_n)


def test_phosphoserine_sidechain_is_dianionic() -> None:
    """The phosphate is deprotonated at pH 7.4, so no hydrogen reaches it."""
    _residue_array, prep = _prepare("phosphopeptide_5ema", "SEP")
    restype = prep.residue_type
    names = {a.name for a in restype.atoms}
    assert {"CB", "OG", "P"} <= names
    phosphate_oxygens = {
        b for a, b, *_ in restype.bonds if a == "P" and b.startswith("O")
    } | {a for a, b, *_ in restype.bonds if b == "P" and a.startswith("O")}
    assert len(phosphate_oxygens) == 4


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_prepare_ligands_routes_polymer_residues(stem: str) -> None:
    """A polymer-linking residue reaches the database with its connections."""
    res_name, net_charge, _n_chi = _FIXTURES[stem]
    param_db, canonical_ordering = prepare_ligands(
        _load(stem), param_db=ParameterDatabase.get_default()
    )

    restype = next(r for r in param_db.chemical.residues if r.name == res_name)
    assert {c.name for c in restype.connections} == {"down", "up"}

    charges = {
        p.atom: p.charge
        for p in param_db.scoring.elec.atom_charge_parameters
        if p.res == res_name
    }
    assert sum(charges.values()) == pytest.approx(net_charge, abs=1e-3)

    # the termini variants apply to it as they do to any polymer residue
    variants = {r.name for r in param_db.chemical.residues if r.base_name == res_name}
    assert {res_name, f"{res_name}:nterm", f"{res_name}:cterm"} <= variants
    assert res_name in canonical_ordering.restype_io_equiv_classes


def test_polymer_residue_requires_a_bond_table() -> None:
    """Chemistry is read from bonds, so a bondless residue is refused."""
    param_db = ParameterDatabase.get_default()
    cif = pdbx.CIFFile.read(str(FIXTURE_DIR / "phosphopeptide_5ema.cif"))
    bondless = pdbx.get_structure(cif, model=1, include_bonds=False)
    with pytest.raises(ValueError, match="bond table"):
        prepare_polymer_residue(
            _residue(bondless, "SEP"),
            rebuild_canonical_ordering(param_db),
            param_db,
        )


def test_unsupported_backbone_is_refused_not_treated_as_a_ligand() -> None:
    """A declared polymer whose atoms match no profile fails loudly."""
    param_db = ParameterDatabase.get_default()
    residue = _residue(_load("phosphopeptide_5ema"), "SEP")
    # drop the backbone carbonyl carbon; nothing left matches a backbone
    residue = residue[residue.atom_name != "C"]
    with pytest.raises(LigandPreparationError, match="no supported backbone"):
        prepare_polymer_residue(residue, rebuild_canonical_ordering(param_db), param_db)


# --------------------------------------------------------------------------- #
# component-type classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ccd_type,expected",
    [
        ("L-PEPTIDE LINKING", True),
        ("D-PEPTIDE LINKING", True),
        ("DNA LINKING", True),
        ("RNA LINKING", True),
        ("D-SACCHARIDE", True),
        ("NON-POLYMER", False),
        ("UNKNOWN", False),
        (None, False),
    ],
)
def test_is_polymer_linking_ccd_type(ccd_type, expected) -> None:
    assert is_polymer_linking_ccd_type(ccd_type) is expected


def test_chem_comp_types_from_cif(tmp_path) -> None:
    """A type declared by the input file classifies a residue the CCD lacks."""
    cif_path = tmp_path / "declared.cif"
    cif_path.write_text(
        "data_test\n"
        "#\n"
        "loop_\n"
        "_chem_comp.id\n"
        "_chem_comp.type\n"
        "ALA 'L-peptide linking'\n"
        f"{UNDEFINED_CODE} 'L-peptide linking'\n"
        "LIG non-polymer\n"
        "#\n"
    )
    types = chem_comp_types_from_cif(cif_path)
    assert types[UNDEFINED_CODE] == "L-PEPTIDE LINKING"
    assert types["LIG"] == "NON-POLYMER"
    assert is_polymer_linking_ccd_type(types[UNDEFINED_CODE])
    assert not is_polymer_linking_ccd_type(types["LIG"])


def test_declared_type_routes_a_residue_the_ccd_does_not_know() -> None:
    """An unrecognized residue name declared as polymer-linking is routed."""
    from tmol.ligand import get_chem_comp_type

    # the declared type is consulted only where the CCD has no answer
    assert get_chem_comp_type(UNDEFINED_CODE) is None

    atom_array = _load("phosphopeptide_5ema")
    renamed = atom_array.copy()
    renamed.res_name[renamed.res_name == "SEP"] = UNDEFINED_CODE

    param_db, _ordering = prepare_ligands(
        renamed,
        param_db=ParameterDatabase.get_default(),
        chem_comp_types={UNDEFINED_CODE: "L-PEPTIDE LINKING"},
    )
    restype = next(r for r in param_db.chemical.residues if r.name == UNDEFINED_CODE)
    assert {c.name for c in restype.connections} == {"down", "up"}
    assert restype.properties.polymer.backbone_type == "alpha"


def test_smiles_path_is_not_routed_to_the_polymer_path() -> None:
    """A SMILES carries no polymer declaration, so it prepares as a ligand.

    Free phosphoserine has an intact alpha-amino-acid backbone, but nothing in
    a SMILES says the molecule is a residue rather than a free molecule. Until
    the SMILES entry point takes an explicit polymer argument, it must keep
    producing a connection-free ligand.
    """
    from tmol.ligand import prepare_ligand_from_smiles

    param_db, _ordering = prepare_ligand_from_smiles(
        "N[C@@H](COP(=O)([O-])[O-])C(=O)[O-]",
        param_db=ParameterDatabase.get_default(),
        res_name="PSR",
        seed=1,
    )
    restype = next(r for r in param_db.chemical.residues if r.name == "PSR")
    assert restype.connections == ()
    assert restype.properties.polymer.is_polymer is False
    # and it is typed as a ligand throughout: no backbone atom types
    assert not {a.atom_type for a in restype.atoms} & {"Nbb", "CAbb", "CObb", "OCbb"}
