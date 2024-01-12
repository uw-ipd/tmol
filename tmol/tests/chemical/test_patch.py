import cattr
import numpy
import yaml
from attrs import evolve

from tmol.chemical.ideal_coords import normalize
from tmol.chemical.restypes import RefinedResidueType
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase

from tmol.database.chemical import VariantType, RawResidueType


def test_patched_residue_construction_smoke(default_database):
    chem_db = default_database.chemical
    ala_rt = next(r for r in chem_db.residues if r.name == "ALA:nterm")

    ala_rrt = cattr.structure(cattr.unstructure(ala_rt), RefinedResidueType)

    assert "down" not in ala_rrt.connection_to_cidx

    upper_conn = ala_rrt.connection_to_cidx["up"]
    assert ala_rrt.atom_downstream_of_conn[upper_conn, 0] == ala_rrt.atom_to_idx["C"]
    assert ala_rrt.atom_downstream_of_conn[upper_conn, 1] == ala_rrt.atom_to_idx["CA"]
    assert ala_rrt.atom_downstream_of_conn[upper_conn, 2] == ala_rrt.atom_to_idx["N"]

    assert ala_rrt.atom_downstream_of_conn.shape == (1, len(ala_rrt.atoms))

    ala_rt = next(r for r in chem_db.residues if r.name == "ALA:cterm:nterm")

    ala_rrt = cattr.structure(cattr.unstructure(ala_rt), RefinedResidueType)

    assert "up" not in ala_rrt.connection_to_cidx
    assert "down" not in ala_rrt.connection_to_cidx


def test_patched_residue_icoor_mapping(default_database):
    chem_db = default_database.chemical
    r = next(r for r in chem_db.residues if r.name == "LEU:nterm")
    leu_rt = cattr.structure(cattr.unstructure(r), RefinedResidueType)

    assert leu_rt.at_to_icoor_ind.shape[0] == leu_rt.n_atoms

    for i, at in enumerate(leu_rt.atoms):
        assert leu_rt.at_to_icoor_ind[i] == leu_rt.icoors_index[at.name]


def test_patched_residue_ideal_coords(default_database):
    chem_db = default_database.chemical
    r = next(r for r in chem_db.residues if r.name == "LEU:cterm")
    leu_rt = cattr.structure(cattr.unstructure(r), RefinedResidueType)

    oxt_ind = leu_rt.icoors_index["OXT"]
    c_ind = leu_rt.icoors_index["C"]
    o_ind = leu_rt.icoors_index["O"]

    oco_ang = numpy.arccos(
        numpy.dot(
            normalize(leu_rt.ideal_coords[oxt_ind, :] - leu_rt.ideal_coords[c_ind, :]),
            normalize(leu_rt.ideal_coords[o_ind, :] - leu_rt.ideal_coords[c_ind, :]),
        )
    )

    oxtc_dis = numpy.linalg.norm(
        leu_rt.ideal_coords[oxt_ind, :] - leu_rt.ideal_coords[c_ind]
    )

    assert abs(oco_ang - (numpy.pi - leu_rt.icoors_geom[oxt_ind, 1])) < 1e-5
    assert abs(oxtc_dis - leu_rt.icoors_geom[oxt_ind, 2]) < 1e-5


def test_patched_pdb(ubq_res):
    assert ubq_res[0].residue_type.name == "MET:nterm"
    assert ubq_res[-1].residue_type.name == "GLY:cterm"


# parse a yaml string as a raw VariantType
def variant_from_yaml(yml_string):
    raw = yaml.safe_load(yml_string)
    return tuple(cattr.structure(x, VariantType) for x in raw)


# parse a yaml string as a raw ResidueType
def residues_from_yaml(yml_string):
    raw = yaml.safe_load(yml_string)
    return tuple(cattr.structure(x, RawResidueType) for x in raw)


def test_uncommon_patching_options(default_unpatched_chemical_database):
    patch = """
  - name:  TestAddConnection
    display_name: addconn
    pattern: 'CC[SH]'
    remove_atoms:
    - <H1>
    add_atoms: []
    add_atom_aliases: []
    modify_atoms: []
    add_connections:
    - { name:  thiol, atom: <S1> }
    add_bonds: []
    icoors:
    - { name: thiol, phi: -180.0 deg, theta: 84.011803 deg, d: 1.329369, parent: <S1>, grand_parent: <C2>, great_grand_parent: <C1>}
    """
    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variant_from_yaml(patch)
    )
    patched_chemdb = PatchedChemicalDatabase.from_chem_db(
        unpatched_chemical_database
    )  # apply patches
    patched_names = [x.name for x in patched_chemdb.residues]
    assert "CYS:addconn" in patched_names

    patch = """
  - name:  TestOddConnection
    display_name: oddconn
    pattern: 'CC[SH]'
    remove_atoms:
    - <H1>
    add_atoms:
    - { name: CDX ,  atom_type: CH3 }
    add_atom_aliases: []
    modify_atoms: []
    add_connections: []
    add_bonds:
    - [  CDX,    <S1>   ]
    icoors:
    - { name: CDX, source: <H1>}
    """
    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variant_from_yaml(patch)
    )
    patched_chemdb = PatchedChemicalDatabase.from_chem_db(
        unpatched_chemical_database
    )  # apply patches
    patched_names = [x.name for x in patched_chemdb.residues]
    assert "CYS:oddconn" in patched_names


def test_patch_error_checks(default_unpatched_chemical_database):
    patch = """
  - name:  TestDuplicateName
    display_name: dupl
    pattern: 'CC[SH]'
    remove_atoms: []
    add_atoms:
    - { name: HG,  atom_type: Hpol }
    add_atom_aliases: []
    modify_atoms: []
    add_connections: []
    add_bonds:
    - [  HG,    <S1>   ]
    icoors:
    - { name: HG, source: <H1>}
    """
    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variant_from_yaml(patch)
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad raw residue: CYS:dupl",
                "Error: duplicated_atom_name; atoms may appear only once",
                "Offending atoms:",
                '    "HG" appears 2 times',
            ]
        )
        assert str(err) == gold_err

        threw = True
    assert threw

    patch = """
  - name:  TestIllegalBond
    display_name: illegal
    pattern: 'CC[SH]'
    remove_atoms:
    - <H1>
    add_atoms:
    - { name: HG2,  atom_type: Hpol }
    add_atom_aliases: []
    modify_atoms: []
    add_connections: []
    add_bonds:
    - [  <H1>,    HG2   ]
    icoors:
    - { name: HG2, source: <H1>}
    """
    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variant_from_yaml(patch)
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad raw residue: CYS:illegal",
                "Error: illegal_bond; must be between declared atoms",
                "Offending atoms:",
                '    Undeclared atom "HG" in bond ("HG", "HG2")',
            ]
        )
        assert str(err) == gold_err
        threw = True
    assert threw


def test_patch_validation_missing_fields(default_unpatched_chemical_database):
    patch = """
  - name:  TestGonnaDropField
    display_name: drop
    pattern: 'CC[SH]'
    remove_atoms: []
    add_atoms: []
    add_atom_aliases: []
    modify_atoms: []
    add_connections: []
    add_bonds: []
    icoors: []
    """
    variants = variant_from_yaml(patch)
    variants[0].modify_atoms = None  # drop a field!
    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variants
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad patch: TestGonnaDropField",
                "Error: Undefined field: modify_atoms",
            ]
        )
        assert str(err) == gold_err

        threw = True
    assert threw

    variants[0].remove_atoms = None  # drop another field!
    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variants
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad patch: TestGonnaDropField",
                "Error: Undefined fields: remove_atoms, modify_atoms",
            ]
        )
        assert str(err) == gold_err

        threw = True
    assert threw


def test_patch_validation_remove_atoms_reference(default_unpatched_chemical_database):
    patch = """
  - name:  BadPatchRemoveAtoms
    display_name: bad
    pattern: 'CC[SH]'
    remove_atoms:
    - HG
    add_atoms: []
    add_atom_aliases: []
    modify_atoms: []
    add_connections: []
    add_bonds: []
    icoors: []
    """

    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variant_from_yaml(patch)
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad patch: BadPatchRemoveAtoms",
                'Error: remove_nonreference_atom; atoms listed with "remove_atoms" must begin with "<" and end with ">".',
                "Offending atoms: HG",
            ]
        )
        assert str(err) == gold_err
        # print(err)

        threw = True
    assert threw


def test_patch_validation_modify_atoms_reference(default_unpatched_chemical_database):
    patch = """
  - name:  BadPatchModifyAtoms
    display_name: bad
    pattern: 'CC[SH]'
    remove_atoms: []
    add_atoms: []
    add_atom_aliases: []
    modify_atoms:
    - {name: CB, atom_type: CAbb}
    add_connections: []
    add_bonds: []
    icoors: []
    """

    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variant_from_yaml(patch)
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad patch: BadPatchModifyAtoms",
                'Error: modify_nonreference_atom; atoms listed with "modify_atoms" must begin with "<" and end with ">".',
                "Offending atoms: CB",
            ]
        )
        assert str(err) == gold_err

        threw = True
    assert threw


def test_patch_validation_illegal_add_alias(default_unpatched_chemical_database):
    patch = """
  - name:  BadPatchAddAlias
    display_name: bad
    pattern: 'CC[SH]'
    remove_atoms: []
    add_atoms: 
    - {name: XX, atom_type: CH3}
    add_atom_aliases:
    - {name: YY, alt_name: ZZ}
    modify_atoms: []
    add_connections: []
    add_bonds: []
    icoors: []
    """

    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variant_from_yaml(patch)
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad patch BadPatchAddAlias",
                "Error: illegal_add_alias. Added atom alias must refer to newly"
                ' added atoms. Bad add_atom_aliases from: "YY" --> "ZZ"',
            ]
        )
        assert str(err) == gold_err

        threw = True
    assert threw


def test_patch_validation_illegal_bond(default_unpatched_chemical_database):
    patch = """
  - name:  BadPatchBond
    display_name: bad
    pattern: 'CC[SH]'
    remove_atoms: []
    add_atoms: 
    - {name: XX, atom_type: CH3}
    add_atom_aliases: []
    modify_atoms: []
    add_connections: []
    add_bonds: 
    - [SG, XX]
    icoors: []
    """

    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variant_from_yaml(patch)
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad patch BadPatchBond",
                'Error: illegal bond; first atom in each bond must be either atom reference (start with "<" and end with ">") or an added atom.',
                'Offending bonds: ("SG" "XX")',
            ]
        )
        assert str(err) == gold_err

        threw = True
    assert threw


def test_patch_validation_illegal_icoor(default_unpatched_chemical_database):
    patch = """
  - name:  BadPatchICoor
    display_name: bad
    pattern: 'CC[SH]'
    remove_atoms: []
    add_atoms: 
    - {name: XX, atom_type: CH3}
    - {name: YY, atom_type: CH3}
    add_atom_aliases: []
    modify_atoms: []
    add_connections: []
    add_bonds: []
    icoors:
    - { name:     XX, source: <H1>, phi:   80.0 deg, theta: 60.0 deg, d: 1.2, parent:  <S1>, grand_parent: <C1>, great_grand_parent: CB}
    - { name:     YY, phi:   80.0 deg, theta: 60.0 deg, d: 1.2, parent:  <S1>, grand_parent: <C1>}
    - { name:     SG, source: <H1>, phi:   80.0 deg, theta: 60.0 deg, d: 1.2, grand_parent: <C1>, great_grand_parent: <C2>}
    """

    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, variants=variant_from_yaml(patch)
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad patch BadPatchICoor",
                'Error: illegal_icoor; icoor atoms must be for either atom reference (start with "<" and end with ">") or an added atom / connection,',
                'and ancestor atoms may only be omitted (i.e. "None") if the "source" atom is provided',
                "Offending icoors:",
                '    Icoor for XX with atom reference "great_grand_parent" of "CB"',
                '    Icoor for YY with atom reference "great_grand_parent" of "None"',
                '    Icoor for SG with atom reference "name" of "SG"',
                'Where the added atom / connection list is: "XX", "YY"',
            ]
        )
        assert str(err) == gold_err

        threw = True
    assert threw


def test_res_error_checks(default_unpatched_chemical_database):
    rawres = """
  - name:  IllegalTorsion
    base_name: IllegalTorsion
    name3: ILL
    io_equiv_class: ILL
    atoms:
    - { name: N   ,  atom_type: Nbb  }
    - { name: CA  ,  atom_type: CAbb }
    - { name: C   ,  atom_type: CObb }
    atom_aliases: []
    bonds:
    - [   N,   CA]
    - [  CA,    C]
    connections:
    - { name:  down, atom: N }
    torsions:
    - name: phi
      a: { connection: down, bond_sep_from_conn: 0 }
      b: { atom: N }
      c: { atom: CA }
      d: { atom: C }
    - name: psi
      a: { atom: N }
      b: { atom: CA }
      c: { atom: C }
      d: { connection: up, bond_sep_from_conn: 0 }
    icoors:
    - { name:     N, phi:    0.000000 deg, theta:    0.000000 deg, d:    0.000000, parent:     N, grand_parent:    CA, great_grand_parent:     C}
    - { name:    CA, phi:    0.000000 deg, theta:  180.000000 deg, d:    1.458001, parent:     N, grand_parent:    CA, great_grand_parent:     C}
    - { name:     C, phi:    0.000000 deg, theta:   68.799995 deg, d:    1.523259, parent:    CA, grand_parent:     N, great_grand_parent:     C}
    - { name:  down, phi: -150.000015 deg, theta:   58.300003 deg, d:    1.328685, parent:     N, grand_parent:    CA, great_grand_parent:     C}
    properties:
      is_canonical: true
      polymer:
        is_polymer: true
        polymer_type: amino_acid
        backbone_type: alpha
        mainchain_atoms:
        - N
        - CA
        - C
        sidechain_chirality: achiral
        termini_variants: []
      chemical_modifications: []
      connectivity: []
      protonation:
        protonated_atoms: []
        protonation_state: neutral
        pH: 7
      virtual: []
    chi_samples: []
    """
    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, residues=residues_from_yaml(rawres)
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad raw residue: IllegalTorsion",
                "Error: illegal_torsion; Torsion atoms must be either previously-declared connections or previously-declared atoms",
                "Offending atom:",
                '    atom "up" of (N, CA, C, up) is not a declared connection',
            ]
        )
        assert str(err) == gold_err

        threw = True
    assert threw


def test_validate_restype_bad_conns(default_unpatched_chemical_database):
    rawres = """
  - name:  ALA
    base_name: ALA
    name3: ALA
    io_equiv_class: ALA
    atoms:
    - { name: N   ,  atom_type: Nbb  }
    - { name: CA  ,  atom_type: CAbb }
    - { name: C   ,  atom_type: CObb }
    - { name: O   ,  atom_type: OCbb }
    - { name: CB  ,  atom_type: CH3  }
    - { name: H   ,  atom_type: HNbb }
    - { name: HA  ,  atom_type: Hapo }
    - { name: HB1 ,  atom_type: Hapo }
    - { name: HB2 ,  atom_type: Hapo }
    - { name: HB3 ,  atom_type: Hapo }
    atom_aliases:
    - { name: HB1 , alt_name: 1HB }
    - { name: HB2 , alt_name: 2HB }
    - { name: HB3 , alt_name: 3HB }
    bonds:
    - [   N,   CA]
    - [   N,    H]
    - [  CA,    C]
    - [   C,    O]
    - [  CA,   CB]
    - [  CA,   HA]
    - [  CB,  HB1]
    - [  CB,  HB2]
    - [  CB,  HB3]
    connections:
    - { name: down, atom: N }
    - { name: up, atom: CX }
    torsions:
    - name: phi
      a: { connection: down, bond_sep_from_conn: 0 }
      b: { atom: N }
      c: { atom: CA }
      d: { atom: C }
    - name: psi
      a: { atom: N }
      b: { atom: CA }
      c: { atom: C }
      d: { connection: up, bond_sep_from_conn: 0 }
    - name: omega
      a: { atom: CA }
      b: { atom: C }
      c: { connection: up, bond_sep_from_conn: 0 }
      d: { connection: up, bond_sep_from_conn: 1 }
    icoors:
    - { name:     N, phi:    0.000000 deg, theta:    0.000000 deg, d:  0.000000, parent:  N, grand_parent: CA, great_grand_parent:    C }
    - { name:    CA, phi:    0.000000 deg, theta:  180.000000 deg, d:  1.458001, parent:  N, grand_parent: CA, great_grand_parent:    C }
    - { name:     C, phi:    0.000000 deg, theta:   68.800003 deg, d:  1.523258, parent: CA, grand_parent:  N, great_grand_parent:    C }
    - { name:    up, phi:  149.999985 deg, theta:   63.800007 deg, d:  1.328685, parent:  C, grand_parent: CA, great_grand_parent:    N }
    - { name:     O, phi: -180.000000 deg, theta:   59.200005 deg, d:  1.231015, parent:  C, grand_parent: CA, great_grand_parent:   up }
    - { name:    CB, phi: -122.800000 deg, theta:   69.625412 deg, d:  1.521736, parent: CA, grand_parent:  N, great_grand_parent:    C }
    - { name:   HB1, phi: -180.000000 deg, theta:   70.500000 deg, d:  1.090040, parent: CB, grand_parent: CA, great_grand_parent:    N }
    - { name:   HB2, phi:  120.000000 deg, theta:   70.500000 deg, d:  1.090069, parent: CB, grand_parent: CA, great_grand_parent:  HB1 }
    - { name:   HB3, phi:  120.000000 deg, theta:   70.500000 deg, d:  1.088803, parent: CB, grand_parent: CA, great_grand_parent:  HB2 }
    - { name:    HA, phi: -119.000000 deg, theta:   71.900000 deg, d:  1.090078, parent: CA, grand_parent:  N, great_grand_parent:   CB }
    - { name:  down, phi: -150.000000 deg, theta:   58.300003 deg, d:  1.328685, parent:  N, grand_parent: CA, great_grand_parent:    C }
    - { name:     H, phi: -180.000000 deg, theta:   60.849998 deg, d:  1.010000, parent:  N, grand_parent: CA, great_grand_parent: down }
    properties:
      is_canonical: true
      polymer:
        is_polymer: true
        polymer_type: amino_acid
        backbone_type: alpha
        mainchain_atoms:
        - N
        - CA
        - C
        sidechain_chirality: l
        termini_variants: []
      chemical_modifications: []
      connectivity: []
      protonation:
        protonated_atoms: []
        protonation_state: neutral
        pH: 7
      virtual: []
    chi_samples: []
"""
    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, residues=residues_from_yaml(rawres)
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad raw residue: ALA",
                "Error: illegal_connection; connection atom must be previously declared",
                "Offending connection:",
                '    connection "up" with undeclared atom "CX"',
            ]
        )
        assert str(err) == gold_err
        threw = True
    assert threw


def test_validate_restype_bad_icoor(default_unpatched_chemical_database):
    rawres = """
  - name:  ALA
    base_name: ALA
    name3: ALA
    io_equiv_class: ALA
    atoms:
    - { name: N   ,  atom_type: Nbb  }
    - { name: CA  ,  atom_type: CAbb }
    - { name: C   ,  atom_type: CObb }
    - { name: O   ,  atom_type: OCbb }
    - { name: CB  ,  atom_type: CH3  }
    - { name: H   ,  atom_type: HNbb }
    - { name: HA  ,  atom_type: Hapo }
    - { name: HB1 ,  atom_type: Hapo }
    - { name: HB2 ,  atom_type: Hapo }
    - { name: HB3 ,  atom_type: Hapo }
    atom_aliases:
    - { name: HB1 , alt_name: 1HB }
    - { name: HB2 , alt_name: 2HB }
    - { name: HB3 , alt_name: 3HB }
    bonds:
    - [   N,   CA]
    - [   N,    H]
    - [  CA,    C]
    - [   C,    O]
    - [  CA,   CB]
    - [  CA,   HA]
    - [  CB,  HB1]
    - [  CB,  HB2]
    - [  CB,  HB3]
    connections:
    - { name: down, atom: N }
    - { name: up, atom: C }
    torsions:
    - name: phi
      a: { connection: down, bond_sep_from_conn: 0 }
      b: { atom: N }
      c: { atom: CA }
      d: { atom: C }
    - name: psi
      a: { atom: N }
      b: { atom: CA }
      c: { atom: C }
      d: { connection: up, bond_sep_from_conn: 0 }
    - name: omega
      a: { atom: CA }
      b: { atom: C }
      c: { connection: up, bond_sep_from_conn: 0 }
      d: { connection: up, bond_sep_from_conn: 1 }
    icoors:
    - { name:     N, phi:    0.000000 deg, theta:    0.000000 deg, d:  0.000000, parent:  N, grand_parent: CA, great_grand_parent:    C }
    - { name:    CA, phi:    0.000000 deg, theta:  180.000000 deg, d:  1.458001, parent:  N, grand_parent: CA, great_grand_parent:    C }
    - { name:     C, phi:    0.000000 deg, theta:   68.800003 deg, d:  1.523258, parent: CA, grand_parent:  N, great_grand_parent:    C }
    - { name:    up, phi:  149.999985 deg, theta:   63.800007 deg, d:  1.328685, parent:  C, grand_parent: CA, great_grand_parent:    N }
    - { name:     O, phi: -180.000000 deg, theta:   59.200005 deg, d:  1.231015, parent:  C, grand_parent: CA, great_grand_parent:   up }
    - { name:    CX, phi: -122.800000 deg, theta:   69.625412 deg, d:  1.521736, parent: CA, grand_parent:  N, great_grand_parent:    C }
    - { name:   HB1, phi: -180.000000 deg, theta:   70.500000 deg, d:  1.090040, parent: CB, grand_parent: CA, great_grand_parent:    N }
    - { name:   HB2, phi:  120.000000 deg, theta:   70.500000 deg, d:  1.090069, parent: CB, grand_parent: CY, great_grand_parent:  HB1 }
    - { name:   HB3, phi:  120.000000 deg, theta:   70.500000 deg, d:  1.088803, parent: CB, grand_parent: CA, great_grand_parent:  HB2 }
    - { name:    HA, phi: -119.000000 deg, theta:   71.900000 deg, d:  1.090078, parent: CA, grand_parent:  N, great_grand_parent:   CB }
    - { name:  down, phi: -150.000000 deg, theta:   58.300003 deg, d:  1.328685, parent:  N, grand_parent: CA, great_grand_parent:    C }
    - { name:     H, phi: -180.000000 deg, theta:   60.849998 deg, d:  1.010000, parent:  N, grand_parent: CA, great_grand_parent: down }
    properties:
      is_canonical: true
      polymer:
        is_polymer: true
        polymer_type: amino_acid
        backbone_type: alpha
        mainchain_atoms:
        - N
        - CA
        - C
        sidechain_chirality: l
        termini_variants: []
      chemical_modifications: []
      connectivity: []
      protonation:
        protonated_atoms: []
        protonation_state: neutral
        pH: 7
      virtual: []
    chi_samples: []
"""
    unpatched_chemical_database = evolve(
        default_unpatched_chemical_database, residues=residues_from_yaml(rawres)
    )
    threw = False
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except RuntimeError as err:
        gold_err = "\n".join(
            [
                "Bad raw residue: ALA",
                "Error: illegal_icoor; must reference previoulsy-declared atoms or connections only.",
                "Offending icoors",
                '    icoor for CX: name atom "CX" undeclared',
                '    icoor for HB2: grand_parent atom "CY" undeclared',
            ]
        )
        assert str(err) == gold_err
        threw = True
    assert threw
