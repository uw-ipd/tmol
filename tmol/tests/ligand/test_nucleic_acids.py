"""Preparation of nucleotides whose backbone is a standard one, and of one that
is not.

A nucleotide is prepared as a polymer residue: the phosphate and the whole
sugar take the canonical nucleic acid atom types and torsions, and only the
base is left to the ligand typer. What hangs off the sugar is not looked at, so
a modified base, a substituted 2' position and a modified phosphate are all
standard backbones and keep the nucleic acid torsion potential.

What is not a standard backbone -- a dinucleotide fused into one component,
say -- is prepared the way a nonstandard peptide is: ligand types throughout,
gen_bonded for geometry, and its own generated 5' and 3' patches.

The assertions are sanity checks rather than goldens: what is pinned is which
class a residue lands in, the backbone it is given, and that the torsions the
nucleic acid term reads are the ones it was calibrated on.
"""

from __future__ import annotations

from collections import defaultdict

import biotite.structure.info as info
import numpy as np
import pytest

from tmol.database import ParameterDatabase
from tmol.ligand import prepare_ligands
from tmol.ligand._polymer_profile import (
    glycosidic_torsion_atoms,
    na_backbone_kind,
    na_profile,
    profile_for_atom_array,
)
from tmol.tests.data import data_path

FIXTURE_DIR = data_path("ncaa_fixtures")

# a residue code no component dictionary defines; most invented three-letter
# codes collide with a real one
_UNDEFINED_CODE = "X_"

# stem -> {residue code: (backbone class, the base its tables come from)}
_FIXTURES: dict[str, dict[str, tuple[str, str | None]]] = {
    "na_dna_5mc_1d17": {"5CM": ("dna", "DC")},
    "na_rna_psu_1bzt": {"PSU": ("rna", "U")},
    "na_rna_2ome_310d": {"OMC": ("rna", "C"), "OMG": ("rna", "G")},
    "na_dna_8og_183d": {"8OG": ("dna", "DG")},
    "na_dna_ttd_1ttd": {"TTD": ("nonstandard_na", None)},
}


def _structure(stem: str):
    from tmol.io import atom_array_from_cif

    return atom_array_from_cif(FIXTURE_DIR / f"{stem}.cif")


def _prepared(structure):
    param_db = ParameterDatabase.get_default()
    known = {r.name for r in param_db.chemical.residues}
    prepared, canonical_ordering = prepare_ligands(structure, param_db=param_db)
    return prepared, known, canonical_ordering


def _component(code: str, drop=()):
    """A CCD component as it appears mid-chain, with its bond table."""
    component = info.residue(code)
    component.res_name[:] = code
    leaving = ["OP3", "HOP3", "HO3'", "HO5'", *drop]
    return component[~np.isin(component.atom_name, leaving)]


def _heavy_adjacency(atom_array):
    element = {str(n): str(e) for n, e in zip(atom_array.atom_name, atom_array.element)}
    adjacency: dict = defaultdict(set)
    for i, j, _order in atom_array.bonds.as_array():
        a, b = str(atom_array.atom_name[i]), str(atom_array.atom_name[j])
        if element[a] == "H" or element[b] == "H":
            continue
        adjacency[a].add(b)
        adjacency[b].add(a)
    return adjacency, element


# --------------------------------------------------------------------------- #
# what counts as a standard backbone
# --------------------------------------------------------------------------- #

# code -> the class its backbone puts it in. Everything but the last keeps a
# five-membered sugar with the mainchain closed on it, however modified the
# base, the 2' position or the phosphate is.
_CLASSES = {
    "DA": "dna",
    "DT": "dna",
    "A": "rna",
    "U": "rna",
    "5CM": "dna",
    "8OG": "dna",
    "PSU": "rna",
    "OMG": "rna",
    "1MA": "rna",
    "3DR": "dna",
    "TTD": None,
}


@pytest.mark.parametrize("code", sorted(_CLASSES))
def test_a_standard_nucleotide_backbone_is_recognized(code: str) -> None:
    """The sugar and the mainchain closed on it, never the residue's name.

    Pseudouridine is the case a name cannot reach and an element test gets
    wrong: its base hangs off the sugar through carbon rather than nitrogen,
    and it is the commonest modified nucleotide in the PDB.
    """
    component = _component(code)
    names = {str(n) for n in component.atom_name}
    connections = frozenset(a for a in ("P", "O3'") if a in names)
    assert connections, code
    assert na_backbone_kind(component, connections) == _CLASSES[code]


def test_a_fused_dinucleotide_is_not_a_standard_backbone() -> None:
    """TTD is two nucleotides in one component, so no single sugar spans it."""
    component = _component("TTD")
    assert {"P", "PB"} <= {str(n) for n in component.atom_name}
    assert na_backbone_kind(component, frozenset({"P", "O3'"})) is None


@pytest.mark.parametrize("code", ["DDG", "DOC"])
def test_a_3_prime_deoxy_nucleotide_ends_its_chain_at_c3(code: str) -> None:
    """A dideoxy chain terminator has no O3': the backbone stops at C3'."""
    profile = profile_for_atom_array(_component(code), frozenset({"P"}))
    assert profile.down == ("down", "P") and profile.up is None
    assert profile.mainchain_atoms[-1] == "C3'"


def test_an_abasic_nucleotide_is_a_backbone_not_a_sugar() -> None:
    """AAB's C1' hydroxyl makes it look like a sugar; its backbone wins."""
    profile = profile_for_atom_array(_component("AAB"), frozenset({"P", "O3'"}))
    assert profile is not None and profile.up == ("up", "O3'")


# --------------------------------------------------------------------------- #
# the glycosidic torsion
# --------------------------------------------------------------------------- #

# code -> the four atoms the nucleic acid torsion tables measure chi by
_CHI = {
    "DA": ("O4'", "C1'", "N9", "C4"),
    "DC": ("O4'", "C1'", "N1", "C2"),
    "PSU": ("O4'", "C1'", "C5", "C4"),
    "8OG": ("O4'", "C1'", "N9", "C4"),
}


@pytest.mark.parametrize("code", sorted(_CHI))
def test_chi_is_measured_the_way_the_tables_were_built(code: str) -> None:
    """A rotatable bond search finds the bond but not which four atoms to use.

    The tables are calibrated on the ring oxygen through the bond to the base:
    the fused carbon in a purine, position 2 in a pyrimidine. Both fall out of
    one rule -- the base neighbour lying in the most rings -- which also gives
    the right answer for a C-glycoside.
    """
    component = _component(code)
    adjacency, element = _heavy_adjacency(component)
    kind = na_backbone_kind(component, frozenset({"P", "O3'"}))
    profile = na_profile(ParameterDatabase.get_default().chemical, kind)
    assert glycosidic_torsion_atoms(profile, adjacency, element) == _CHI[code]


# --------------------------------------------------------------------------- #
# whole structures
# --------------------------------------------------------------------------- #

_NA_BACKBONE_TYPES = frozenset({"Pdna", "OOP", "Oet2", "Oet3", "CH1", "CH2", "Hapo"})


@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_a_nucleotide_prepares_from_a_structure(stem: str) -> None:
    """Detection reads the backbone off the bonds; nothing declares it."""
    prepared, known, canonical_ordering = _prepared(_structure(stem))
    added = {
        r.name: r
        for r in prepared.chemical.residues
        if r.name not in known and ":" not in r.name
    }
    expected = _FIXTURES[stem]
    assert set(added) == set(expected), stem

    for code, (backbone, base_reference) in expected.items():
        restype = added[code]
        assert restype.properties.polymer.backbone_type == backbone, code
        assert restype.na_base_reference == base_reference, code
        assert {c.name for c in restype.connections} == {"down", "up"}, code
        assert code in canonical_ordering.restype_io_equiv_classes


@pytest.mark.parametrize(
    "stem",
    [
        s
        for s, v in _FIXTURES.items()
        if all(c[0] != "nonstandard_na" for c in v.values())
    ],
)
def test_a_standard_nucleotide_keeps_the_backbone_and_its_torsions(stem: str) -> None:
    """Nucleic acid types on the phosphate and sugar, ligand types on the base.

    The torsion term reads delta, the sugar puckers and chi off the residue, so
    a backbone that does not declare them is scored on its backbone alone.
    """
    prepared, known, _co = _prepared(_structure(stem))
    for code in _FIXTURES[stem]:
        restype = next(r for r in prepared.chemical.residues if r.name == code)
        types = {a.name: a.atom_type for a in restype.atoms}
        mainchain = restype.properties.polymer.mainchain_atoms
        assert mainchain == ("P", "O5'", "C5'", "C4'", "C3'", "O3'"), code

        # the sugar is backbone too, not only the atoms the mainchain runs through
        for atom in (*mainchain, "O4'", "C1'", "C2'", "OP1", "OP2"):
            assert types[atom] in _NA_BACKBONE_TYPES, (code, atom, types[atom])

        declared = {t.name for t in restype.torsions}
        assert {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"} <= declared
        assert {"nu0", "nu1", "nu4", "chi1"} <= declared


def test_a_fused_dinucleotide_is_prepared_as_a_ligand() -> None:
    """No sugar spans its backbone, so no nucleic acid torsion describes it."""
    prepared, known, _co = _prepared(_structure("na_dna_ttd_1ttd"))
    restype = next(r for r in prepared.chemical.residues if r.name == "TTD")

    assert restype.properties.polymer.backbone_type == "nonstandard_na"
    assert restype.na_base_reference is None
    # both phosphates lie on its mainchain, which is why it is not standard
    mainchain = restype.properties.polymer.mainchain_atoms
    assert {"P", "PB"} <= set(mainchain)
    types = {a.atom_type for a in restype.atoms}
    assert not types & _NA_BACKBONE_TYPES

    # and it brings its own chain-end patches, as a nonstandard peptide does
    variants = {
        r.name.partition(":")[2]
        for r in prepared.chemical.residues
        if r.name.startswith("TTD:")
    }
    assert {"na5prime", "na3prime"} <= variants


def test_every_canonical_base_identifies_as_itself() -> None:
    """The base reference is inferred from chemistry, so it must be a fixpoint."""
    from tmol.ligand._polymer_profile import na_base_reference, na_profile

    chemdb = ParameterDatabase.get_default().chemical
    canonicals = [
        r
        for r in chemdb.residues
        if r.name == r.base_name and r.properties.polymer.polymer_type == "nucleic_acid"
    ]
    assert canonicals
    for residue in canonicals:
        kind = residue.properties.polymer.backbone_type
        inferred = na_base_reference(residue, na_profile(chemdb, kind), chemdb)
        assert inferred == residue.io_equiv_class, residue.name


@pytest.mark.parametrize(
    "stem,code,base",
    [
        ("na_dna_5mc_1d17", "5CM", "DC"),
        ("na_dna_8og_183d", "8OG", "DG"),
        ("na_rna_psu_1bzt", "PSU", "U"),
        ("na_rna_2ome_310d", "OMC", "C"),
        ("na_rna_2ome_310d", "OMG", "G"),
    ],
)
def test_a_modified_base_is_scored_on_the_base_it_modifies(
    stem: str, code: str, base: str
) -> None:
    """Which tables a modified nucleotide takes comes from its own chemistry.

    The base skeleton is matched against the canonical bases of the same
    polymer and the closest wins, so nothing has to declare what the residue is
    a modification of -- 8-oxoguanine keeps guanine's tables because guanine is
    still the nearest of the four.
    """
    prepared, _known, _co = _prepared(_structure(stem))
    restype = next(r for r in prepared.chemical.residues if r.name == code)
    assert restype.na_base_reference == base


def test_a_nucleotide_seen_only_at_a_terminus_gets_its_phosphate_back() -> None:
    """A 5'-terminal residue has no phosphate, but its residue type needs one.

    The type has to describe the residue in a chain, so the phosphate comes
    from the canonical nucleotide of its class, placed by superimposing that
    residue on this one. The terminal form is then the na5prime variant, which
    takes the phosphate off again where the residue really does sit at a 5'
    end. The residue code is one no dictionary defines: the backbone is read
    off the sugar and the base off its own skeleton.
    """
    structure = _structure("na_dna_8og_183d")
    structure.res_name[structure.res_name == "8OG"] = _UNDEFINED_CODE
    stripped = np.isin(structure.atom_name, ["P", "OP1", "OP2"])
    structure = structure[~((structure.res_name == _UNDEFINED_CODE) & stripped)]

    prepared, _known, _co = _prepared(structure)
    restype = next(r for r in prepared.chemical.residues if r.name == _UNDEFINED_CODE)

    assert restype.properties.polymer.backbone_type == "dna"
    assert {"P", "OP1", "OP2"} <= {a.name for a in restype.atoms}
    assert {c.name for c in restype.connections} == {"down", "up"}
    # the base is identified from its own skeleton, so a code nothing defines
    #    is still scored on guanine rather than on the averaged base
    assert restype.na_base_reference == "DG"

    variants = {
        r.name.partition(":")[2]
        for r in prepared.chemical.residues
        if r.name.startswith(f"{_UNDEFINED_CODE}:")
    }
    assert "na5prime" in variants


@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_a_nucleotide_scores(stem: str, torch_device) -> None:
    """Smoke test: preparation has to survive into a pose and a number."""
    from tmol.io import pose_stack_from_biotite
    from tmol.score import beta2016_score_function

    structure = _structure(stem)
    prepared, _known, _co = _prepared(structure)
    pose_stack = pose_stack_from_biotite(structure, torch_device, param_db=prepared)
    sfxn = beta2016_score_function(torch_device, param_db=prepared)
    module = sfxn.render_whole_pose_scoring_module(pose_stack)
    total = float(module(pose_stack.coords).sum())

    assert np.isfinite(total), stem
    assert total != 0.0, stem


@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_every_variant_carries_charges_for_all_of_its_atoms(stem: str) -> None:
    """A missing charge fails the whole structure, used variant or not.

    PackedBlockTypes packs every variant of every residue it is given, so a
    5' phosphate form nothing sits in still has to resolve.
    """
    prepared, known, _co = _prepared(_structure(stem))
    charges: dict = {}
    for entry in prepared.scoring.elec.atom_charge_parameters:
        res, _, variant = str(entry.res).partition(":")
        charges.setdefault(res, {}).setdefault(str(entry.atom), {})[
            variant
        ] = entry.charge

    missing = []
    for restype in prepared.chemical.residues:
        if restype.name in known:
            continue
        base_name, *variants = restype.name.split(":")
        variants.append("")  # unpatched last, as the elec resolver does
        for atom in restype.atoms:
            by_variant = charges.get(base_name, {}).get(atom.name)
            if not any(v in (by_variant or {}) for v in variants):
                missing.append(f"{restype.name}/{atom.name}")
    assert missing == []


@pytest.mark.parametrize("stem", sorted(_FIXTURES))
def test_every_variant_builds_finite_ideal_coordinates(stem: str) -> None:
    """A frame left pointing at an atom a patch removed builds NaN.

    Every terminal form is built here, not only the ones this structure uses,
    for the same reason their charges are: they all get packed.
    """
    import cattr

    from tmol.chemical._restypes import RefinedResidueType

    prepared, known, _co = _prepared(_structure(stem))
    bad = []
    for restype in prepared.chemical.residues:
        if restype.name in known:
            continue
        refined = cattr.structure(cattr.unstructure(restype), RefinedResidueType)
        if not np.isfinite(refined.compute_ideal_coords()).all():
            bad.append(restype.name)
    assert bad == []


def _sweep_structure(stem: str):
    from tmol.io import atom_array_from_cif

    return atom_array_from_cif(data_path("sweep_regressions", f"{stem}.cif.zst"))


def _block_type_names(structure, device):
    from tmol.io import pose_stack_from_biotite

    pose = pose_stack_from_biotite(
        structure, device, prepare_ligands=True, ligand_seed=0, no_optH=True
    )
    types = pose.packed_block_types.active_block_types
    names = [types[i].name for i in pose.block_type_ind64[0].tolist() if i >= 0]
    return names, pose.packed_block_types.chem_db


def test_a_substituted_five_prime_oxygen_leaves_no_five_prime_port() -> None:
    """1CX5 A:7 MMT's 5' oxygen bonds the methylimino link to the previous residue.

    No phosphate is grafted onto that ether: the type keeps only its 3' port,
    and the link at C3X is a conjugation.
    """
    prepared, _known, _co = _prepared(_sweep_structure("mmt_5prime_ether_1cx5"))
    types = {r.name: r for r in prepared.chemical.residues if r.base_name == "MMT"}

    assert "P" not in {a.name for a in types["MMT"].atoms}
    assert {c.name for c in types["MMT"].connections} == {"up"}
    assert "MMT:conj_C3X" in types


def test_a_substituted_three_prime_oxygen_leaves_no_three_prime_port(
    torch_device,
) -> None:
    """CCC's 3' oxygen closes its 2',3'-cyclic phosphate (1hq1 B178), so the
    chain-end copy is no 3' terminus: nothing puts a hydrogen on that O3'."""
    names, _chemdb = _block_type_names(
        _sweep_structure("cyclic_phosphate_3prime_1hq1"), torch_device
    )

    assert names == ["RC:na5primephos", "CCC"]


def test_the_component_definition_breaks_a_leaving_oxygen_tie(torch_device) -> None:
    """LCC's 5'-terminal copy is completed with the definition's O1P and OXT.

    Both are terminal hydroxyls on P; the definition declares OXT as leaving,
    so O1P stays, as in the copies inside the chain (6C8D A:1-A:3).
    """
    structure = _sweep_structure("lcc_leaving_atoms_6c8d")
    names, _chemdb = _block_type_names(structure, torch_device)

    assert names == ["LCC:na5prime", "LCC", "LCC", "LCG", "RA:na3prime"]


def test_a_patch_removes_the_hydrogens_of_the_atoms_it_removes() -> None:
    """XY7's phosphate carries hydroxyl hydrogens a canonical nucleotide lacks.

    The 5'-terminal patches remove OP1 and OP2, and with them HOP1 and HOP2 (7KW4
    A:2).
    """
    prepared, _known, _co = _prepared(_sweep_structure("xy7_phosphate_hydrogens_7kw4"))
    atoms = {
        r.name: {a.name for a in r.atoms}
        for r in prepared.chemical.residues
        if r.base_name == "XY7"
    }

    assert {"HOP1", "HOP2"} <= atoms["XY7"]
    assert not {"HOP1", "HOP2"} & (atoms["XY7:na5prime"] | atoms["XY7:na5primephos"])


def test_a_variant_keeps_the_phosphate_alias_of_its_residue(torch_device) -> None:
    """1MWI D:7 AAB reads its deposited O3P as OP2; so do its variants, na5primephos
    too."""
    structure = _sweep_structure("aab_phosphate_alias_1mwi")
    names, chemdb = _block_type_names(structure, torch_device)

    assert names == ["DG:na5primephos", "AAB", "DT:na3prime"]
    aab = [r for r in chemdb.residues if r.base_name == "AAB"]
    assert len(aab) > 1
    for restype in aab:
        aliases = {a.alt_name: a.name for a in restype.atom_aliases}
        assert aliases["O3P"] == "OP2", restype.name
