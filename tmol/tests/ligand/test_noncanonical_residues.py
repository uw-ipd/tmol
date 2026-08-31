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
    is_polymer_linking_component_type,
    prepare_ligands,
    prepare_polymer_residue,
)
from tmol.ligand._preparation import _ideal_coords_by_name
from tmol.ligand._polymer_profile import alpha_profile
from tmol.ligand._registry import rebuild_canonical_ordering
from tmol.tests.data import data_path

FIXTURE_DIR = data_path("ncaa_fixtures")

# A component id no dictionary defines
UNDEFINED_CODE = "X_"

# stem -> (residue code, net formal charge at pH 7.4, expected chi count)
_FIXTURES: dict[str, tuple[str, float, int]] = {
    "phosphopeptide_5ema": ("SEP", -2.0, 3),
    "collagen_hyp_1bkv": ("HYP", 0.0, 1),
}


def _alpha():
    """The alpha profile, as derived from the default database."""
    return alpha_profile(ParameterDatabase.get_default().chemical)


def _load(stem: str) -> struc.AtomArray:
    """Read a fixture with its bond table, as the polymer path requires."""
    cif = pdbx.CIFFile.read(str(FIXTURE_DIR / f"{stem}.cif"))
    return pdbx.get_structure(
        cif, model=1, include_bonds=True, extra_fields=["label_seq_id"]
    )


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


def _prepare(stem: str, res_name: str, connection_atoms=None):
    param_db = ParameterDatabase.get_default()
    residue = _residue(_load(stem), res_name)
    prep = prepare_polymer_residue(
        residue,
        rebuild_canonical_ordering(param_db),
        param_db,
        connection_atoms=connection_atoms,
    )
    return residue, prep


def _assert_alpha_amino_acid(residue: struc.AtomArray, prep) -> None:
    """Every invariant an alpha-amino-acid residue type must satisfy."""
    restype = prep.residue_type
    names = {a.name for a in restype.atoms}

    assert len(names) == len(restype.atoms), "duplicate atom names"

    # the caps are scaffolding; none of them may survive into the residue type
    assert names.isdisjoint(_alpha().cap_names)

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
    assert polymer.backbone_type == "alpha_aa"
    assert polymer.sidechain_chirality == "l"
    assert tuple(polymer.mainchain_atoms) == _alpha().mainchain_atoms

    types = {a.name: a.atom_type for a in restype.atoms}
    assert types["CA"] == "CAbb"
    assert types["C"] == "CObb"
    assert types["O"] == "OCbb"
    assert types["N"] in _alpha().amide_n_types

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


def test_a_residue_seen_only_at_a_terminus_needs_no_ccd_entry() -> None:
    """One connection does not say where the backbone stops; chemistry does.

    A residue at the end of a chain bonds on one side only, and its own
    carbonyl carries a single oxygen because the terminal hydroxyl was never
    A component definition would name both chain ends, so the same residue
    under a code nothing defines has to reach the same backbone without one.
    """
    param_db = ParameterDatabase.get_default()
    structure = _load("phosphopeptide_5ema")
    last = max(int(i) for i in structure.res_id[structure.res_name == "SEP"])
    structure = structure[structure.res_id <= last]
    structure.res_name[structure.res_name == "SEP"] = "XEP"

    prepared, _ordering = prepare_ligands(structure, param_db=param_db, seed=1234)
    restype = next(r for r in prepared.chemical.residues if r.name == "XEP")

    assert restype.properties.polymer.backbone_type == "alpha_aa"
    assert restype.properties.polymer.mainchain_atoms == ("N", "CA", "C")
    assert {c.name for c in restype.connections} == {"down", "up"}
    variants = {r.name for r in prepared.chemical.residues if r.base_name == "XEP"}
    assert {"XEP:nterm", "XEP:cterm"} <= variants


def test_preparation_is_reproducible_under_a_fixed_seed() -> None:
    """A random conformer makes the residue type differ between runs."""
    param_db = ParameterDatabase.get_default()

    def icoors(seed):
        prepared, _ordering = prepare_ligands(
            _load("phosphopeptide_5ema"), param_db=param_db, seed=seed
        )
        restype = next(r for r in prepared.chemical.residues if r.name == "SEP")
        return {i.name: (i.d, i.theta, i.phi) for i in restype.icoors}

    assert icoors(1234) == icoors(1234)


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
    "component_type,expected",
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
def test_is_polymer_linking_component_type(component_type, expected) -> None:
    assert is_polymer_linking_component_type(component_type) is expected


def test_chem_comp_types_from_cif(tmp_path) -> None:
    """A type declared by the input file classifies a residue by itself."""
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
    assert is_polymer_linking_component_type(types[UNDEFINED_CODE])
    assert not is_polymer_linking_component_type(types["LIG"])


def test_declared_type_routes_an_unrecognized_residue() -> None:
    """An unrecognized residue name declared as polymer-linking is routed."""
    from tmol.ligand import get_chem_comp_type

    # nothing is declared for it unless the input file says so
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
    assert restype.properties.polymer.backbone_type == "alpha_aa"


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


# --------------------------------------------------------------------------- #
# terminal caps
# --------------------------------------------------------------------------- #

CAP_FIXTURE = "capped_peptide_ace_nme"

# residue code -> (connection, atom it attaches to, the atoms of the peptide
#                  bond it makes, which are typed as backbone)
_CAPS = {
    "ACE": ("up", "C", {"C": "CObb", "O": "OCbb"}),
    "NME": ("down", "N", {"N": "Nbb"}),
}


@pytest.mark.parametrize("res_name", sorted(_CAPS))
def test_prepare_peptide_cap(res_name: str) -> None:
    """A cap prepares to a residue type with a single polymer connection."""
    conn_name, conn_atom, types = _CAPS[res_name]
    # a cap is recognized by having exactly one connection, so it cannot be
    #    identified without knowing where that connection is
    residue, prep = _prepare(CAP_FIXTURE, res_name, frozenset({conn_atom}))
    restype = prep.residue_type

    assert {c.name: c.atom for c in restype.connections} == {conn_name: conn_atom}

    # every heavy atom of the input survives; the stubs used to cap it do not
    heavy = {a.name for a in restype.atoms if not a.name.startswith("H")}
    assert heavy == {str(n) for n in residue.atom_name if not str(n).startswith("H")}

    # the peptide bond a cap makes is real, so its atoms are typed as backbone
    assert {a.name: a.atom_type for a in restype.atoms if a.name in types} == types

    # no sidechain, so no chi and no handedness
    assert [t.name for t in restype.torsions] == []
    assert restype.properties.polymer.sidechain_chirality == "achiral"
    assert restype.properties.polymer.is_polymer

    # the methyl is protonated back to a full sp3 carbon
    methyl = "CH3" if res_name == "ACE" else "C"
    bonded = [b for b in restype.bonds if methyl in b[:2]]
    assert sum(1 for b in bonded if any(x.startswith("H") for x in b[:2])) == 3

    # the connection pseudo-atom is placed alongside the real atoms
    coords = _ideal_coords_by_name(restype)
    assert set(coords) == {a.name for a in restype.atoms} | {conn_name}
    assert all(np.all(np.isfinite(xyz)) for xyz in coords.values())


NH2_FIXTURE = "capped_peptide_ace_nh2"


def test_prepare_single_atom_cap() -> None:
    """A cap with one heavy atom has no frame of its own to build on.

    NH2 is a bare amide nitrogen, so the stubs that complete it are placed
    against invented reference points rather than against its neighbours.
    """
    _residue_arr, prep = _prepare(NH2_FIXTURE, "NH2", frozenset({"N"}))
    restype = prep.residue_type

    assert {c.name: c.atom for c in restype.connections} == {"down": "N"}
    assert [a.name for a in restype.atoms if not a.name.startswith("H")] == ["N"]
    assert restype.properties.polymer.is_polymer

    # the peptide bond it makes is real, so its nitrogen is typed as backbone
    n_type = next(a.atom_type for a in restype.atoms if a.name == "N")
    assert n_type in _alpha().amide_n_types

    # and it is protonated back to a full amide
    assert sum(1 for b in restype.bonds if any(x.startswith("H") for x in b[:2])) == 2

    coords = _ideal_coords_by_name(restype)
    assert set(coords) == {a.name for a in restype.atoms} | {"down"}
    assert all(np.all(np.isfinite(xyz)) for xyz in coords.values())


def test_peptide_cap_charges_balance_across_the_pair() -> None:
    """Each cap keeps its share of the amide bond it was cut from.

    Both caps model as n-methylacetamide -- one supplies the acetyl, the other
    the methylamide -- so the charge transferred across the amide C-N stays with
    whichever half retained the atom. Neither half is integral on its own; the
    two are equal and opposite.
    """
    nets = {}
    for res_name, (_conn_name, conn_atom, _types) in _CAPS.items():
        _residue_arr, prep = _prepare(CAP_FIXTURE, res_name, frozenset({conn_atom}))
        nets[res_name] = sum(prep.partial_charges.values())

    assert sum(nets.values()) == pytest.approx(0.0, abs=1e-4)
    assert all(abs(net) < 0.1 for net in nets.values()), nets


def test_prepare_ligands_routes_peptide_caps() -> None:
    """A cap belongs to the chain's entity, which is what routes it."""
    param_db, canonical_ordering = prepare_ligands(
        _load(CAP_FIXTURE), param_db=ParameterDatabase.get_default()
    )
    for res_name, (conn_name, conn_atom, _types) in _CAPS.items():
        restype = next(r for r in param_db.chemical.residues if r.name == res_name)
        assert {c.name: c.atom for c in restype.connections} == {conn_name: conn_atom}
        assert res_name in canonical_ordering.restype_io_equiv_classes


def test_an_unlinked_cap_is_not_routed_to_the_polymer_path() -> None:
    """Only a cap that is actually bonded into a chain is a chain member."""
    from tmol.ligand._preparation import _routes_to_polymer_path
    from tmol.ligand._detect import detect_nonstandard_residues

    param_db = ParameterDatabase.get_default()
    canonical_ordering = rebuild_canonical_ordering(param_db)
    linked = _load(CAP_FIXTURE)
    detected = {
        lig.res_name: lig
        for lig in detect_nonstandard_residues(linked, canonical_ordering)
    }
    assert all(_routes_to_polymer_path(lig) for lig in detected.values())

    # the same caps with nothing to bond to: neither the declared linkages nor
    #    a bond length away from the chain, so they are free molecules whatever
    #    the file's entity says about them
    cif = pdbx.CIFFile.read(str(FIXTURE_DIR / f"{CAP_FIXTURE}.cif"))
    del cif[next(iter(cif.keys()))]["struct_conn"]
    unlinked = pdbx.get_structure(
        cif, model=1, include_bonds=True, extra_fields=["label_seq_id"]
    )
    adrift = np.isin(unlinked.res_name, sorted(_CAPS))
    unlinked.coord[adrift] += 50.0
    for lig in detect_nonstandard_residues(unlinked, canonical_ordering):
        assert not lig.covalently_linked, lig.res_name
        assert not _routes_to_polymer_path(lig), lig.res_name


def test_a_linked_fragment_is_not_mistaken_for_a_cap() -> None:
    """Cap profiles match their whole atom set, not a subset of it."""
    from tmol.ligand._polymer_profile import profile_for_atom_array

    residue = _residue(_load("phosphopeptide_5ema"), "SEP")
    # SEP carries both of NME's atoms, and more besides
    assert {"N", "C"} <= set(residue.atom_name)
    assert profile_for_atom_array(residue) is _alpha()

    # a two-atom fragment whose atoms are not the cap's is not claimed either
    fragment = residue[np.isin(residue.atom_name, ["N", "CA"])]
    assert profile_for_atom_array(fragment) is None


def test_a_cap_carries_the_peptide_bond_terms_its_names_would_miss() -> None:
    """Cartbonded reaches across the peptide bond by atom name.

    A cap does not use a backbone's names -- an acetyl's alpha-equivalent is
    its methyl, a methylamide's is the carbon those rows call CN -- so the
    terms spanning its peptide bond are passed over. It carries copies under
    the names it does use, values unchanged.
    """
    expected = {
        "ACE": {
            ("CH3", "C", "+N"),
            ("CH3", "C", "+N", "+H"),
            ("CH3", "C", "+N", "+CN"),
        },
        "NME": {("C", "N", "+C", "+O"), ("C", "N", "+C", "+CA")},
    }
    for res_name, (_conn_name, conn_atom, _types) in _CAPS.items():
        _residue_arr, prep = _prepare(CAP_FIXTURE, res_name, frozenset({conn_atom}))
        params = prep.cartbonded_params
        rows = {
            tuple(str(a) for a in atoms)
            for atoms in (
                *((p.atm1, p.atm2) for p in params.length_parameters),
                *((p.atm1, p.atm2, p.atm3) for p in params.angle_parameters),
                *((p.atm1, p.atm2, p.atm3, p.atm4) for p in params.torsion_parameters),
            )
            if any(str(a).startswith("+") for a in atoms)
        }
        assert rows == expected[res_name], res_name
