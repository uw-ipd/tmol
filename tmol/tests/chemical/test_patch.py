import cattr
import numpy

from tmol.chemical.ideal_coords import normalize
from tmol.chemical.restypes import RefinedResidueType


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
