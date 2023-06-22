import pytest
import cattr
import numpy

from tmol.chemical.ideal_coords import normalize, build_ideal_coords
from tmol.chemical.restypes import (
    RefinedResidueType,
    ResidueTypeSet,
    find_simple_polymeric_connections,
)
from tmol.tests.data.pdb import data as test_pdbs
from tmol.system.io import read_pdb
from tmol.system.packed import PackedResidueSystem

test_names = ["1QYS", "1UBQ"]


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
