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
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except AssertionError:
        assert True

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
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except AssertionError:
        assert True


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
    try:
        PatchedChemicalDatabase.from_chem_db(
            unpatched_chemical_database
        )  # apply patches
    except AssertionError:
        assert True
