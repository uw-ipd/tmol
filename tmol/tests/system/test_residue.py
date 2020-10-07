import pytest
import cattr

from tmol.system.restypes import RefinedResidueType, ResidueTypeSet
from tmol.tests.data.pdb import data as test_pdbs
from tmol.system.io import read_pdb
from tmol.system.packed import PackedResidueSystem

test_names = ["1QYS", "1UBQ"]


@pytest.mark.parametrize("structure", test_names)
def test_smoke_io(structure):

    for tname in test_names:
        pdb = test_pdbs[tname]
        read_pdb(pdb)


def test_water_system(water_box_res):
    water_system = PackedResidueSystem.from_residues(water_box_res)

    nwat = len(water_box_res)

    assert len(water_system.residues) == nwat

    assert water_system.block_size > 3
    assert len(water_system.atom_metadata) == nwat * water_system.block_size

    assert len(water_system.torsion_metadata) == 0
    assert len(water_system.connection_metadata) == 0


def test_refined_residue_construction_smoke(default_database):
    chem_db = default_database.chemical
    ala_rt = next(r for r in chem_db.residues if r.name == "ALA")

    ala_rrt = cattr.structure(cattr.unstructure(ala_rt), RefinedResidueType)

    lower_conn = ala_rrt.connection_to_cidx["down"]
    assert ala_rrt.atom_downstream_of_conn[lower_conn, 0] == ala_rrt.atom_to_idx["N"]
    assert ala_rrt.atom_downstream_of_conn[lower_conn, 1] == ala_rrt.atom_to_idx["CA"]
    assert ala_rrt.atom_downstream_of_conn[lower_conn, 2] == ala_rrt.atom_to_idx["C"]
    upper_conn = ala_rrt.connection_to_cidx["up"]
    assert ala_rrt.atom_downstream_of_conn[upper_conn, 0] == ala_rrt.atom_to_idx["C"]
    assert ala_rrt.atom_downstream_of_conn[upper_conn, 1] == ala_rrt.atom_to_idx["CA"]
    assert ala_rrt.atom_downstream_of_conn[upper_conn, 2] == ala_rrt.atom_to_idx["N"]

    assert ala_rrt.atom_downstream_of_conn.shape == (2, len(ala_rrt.atoms))


def test_residue_type_set_construction(default_database):
    restype_set = ResidueTypeSet.from_database(default_database.chemical)
    for rt in restype_set.residue_types:
        assert rt in restype_set.restype_map[rt.name3]

    allnames = set([rt.name for rt in restype_set.residue_types])
    for rt in default_database.chemical.residues:
        assert rt.name in allnames
