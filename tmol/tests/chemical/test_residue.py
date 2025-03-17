import cattr
import numpy

from tmol.chemical.ideal_coords import normalize, build_ideal_coords
from tmol.chemical.restypes import (
    RefinedResidueType,
    ResidueTypeSet,
)


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


def test_refined_residue_icoor_mapping(default_database):
    chem_db = default_database.chemical
    r = next(r for r in chem_db.residues if r.name == "LEU")
    leu_rt = cattr.structure(cattr.unstructure(r), RefinedResidueType)

    assert leu_rt.at_to_icoor_ind.shape[0] == leu_rt.n_atoms

    for i, at in enumerate(leu_rt.atoms):
        assert leu_rt.at_to_icoor_ind[i] == leu_rt.icoors_index[at.name]


def test_refined_residue_ideal_coords(default_database):
    chem_db = default_database.chemical
    r = next(r for r in chem_db.residues if r.name == "LEU")
    leu_rt = cattr.structure(cattr.unstructure(r), RefinedResidueType)

    n_ind = leu_rt.icoors_index["N"]
    ca_ind = leu_rt.icoors_index["CA"]
    cb_ind = leu_rt.icoors_index["CB"]

    cb_ang = numpy.arccos(
        numpy.dot(
            normalize(leu_rt.ideal_coords[n_ind, :] - leu_rt.ideal_coords[ca_ind, :]),
            normalize(leu_rt.ideal_coords[cb_ind, :] - leu_rt.ideal_coords[ca_ind, :]),
        )
    )

    cb_dis = numpy.linalg.norm(
        leu_rt.ideal_coords[ca_ind, :] - leu_rt.ideal_coords[cb_ind]
    )

    assert abs(cb_ang - (numpy.pi - leu_rt.icoors_geom[cb_ind, 1])) < 1e-5
    assert abs(cb_dis - leu_rt.icoors_geom[cb_ind, 2]) < 1e-5


def test_refined_residue_ordered_torsions(default_database):
    chem_db = default_database.chemical
    r = next(r for r in chem_db.residues if r.name == "LEU")
    leu_rt = cattr.structure(cattr.unstructure(r), RefinedResidueType)

    assert leu_rt.ordered_torsions.shape == (len(r.torsions), 4, 3)
    assert leu_rt.ordered_torsions.dtype == numpy.int32

    for i in range(leu_rt.ordered_torsions.shape[0]):
        for j in range(4):
            numpy.testing.assert_equal(
                leu_rt.ordered_torsions[i, j],
                numpy.array(
                    leu_rt.torsion_to_uaids[r.torsions[i].name][j], dtype=numpy.int32
                ),
            )


def test_residue_type_set_construction(default_database):
    restype_set = ResidueTypeSet.from_database(default_database.chemical)
    for rt in restype_set.residue_types:
        assert rt in restype_set.restype_map[rt.name3]

    allnames = set([rt.name for rt in restype_set.residue_types])
    for rt in default_database.chemical.residues:
        assert rt.name in allnames


def test_residue_type_set_get_default():
    restype_set1 = ResidueTypeSet.get_default()
    restype_set2 = ResidueTypeSet.get_default()
    assert restype_set1 is restype_set2


def test_build_ideal_coords_smoke(default_database):
    restype_set = ResidueTypeSet.from_database(default_database.chemical)
    for res in restype_set.residue_types:
        build_ideal_coords(res)


def test_all_bonds_construction(fresh_default_restype_set):
    for bt in fresh_default_restype_set.residue_types:
        assert bt.all_bonds.shape[1] == 3
        for i in range(len(bt.all_bonds)):
            at1 = bt.all_bonds[i, 0]
            assert at1 >= 0
            assert at1 <= bt.n_atoms
            assert i >= bt.atom_all_bond_ranges[at1, 0]
            assert i < bt.atom_all_bond_ranges[at1, 1]
            at2 = bt.all_bonds[i, 1]
            if at2 >= 0:
                assert at2 < bt.n_atoms
                assert bt.all_bonds[i, 2] == -1
            else:
                conn = bt.all_bonds[i, 2]
                assert bt.ordered_connection_atoms[conn] == at1


def test_mc_sc_torsion_properties(fresh_default_restype_set):
    leu_restype = next(
        x for x in fresh_default_restype_set.residue_types if x.name == "LEU"
    )
    assert leu_restype.n_mc_torsions == 3
    assert leu_restype.n_sc_torsions == 2
    assert leu_restype.is_torsion_mc[0]
    assert leu_restype.is_torsion_mc[1]
    assert leu_restype.is_torsion_mc[2]
    assert leu_restype.is_torsion_mc[3].item() is False
    assert leu_restype.is_torsion_mc[4].item() is False
    assert leu_restype.mc_torsions[0] == 0
    assert leu_restype.mc_torsions[1] == 1
    assert leu_restype.mc_torsions[2] == 2
    assert leu_restype.sc_torsions[0] == 3
    assert leu_restype.sc_torsions[1] == 4
    assert leu_restype.which_mcsc_torsion[0] == 0
    assert leu_restype.which_mcsc_torsion[1] == 1
    assert leu_restype.which_mcsc_torsion[2] == 2
    assert leu_restype.which_mcsc_torsion[3] == 0
    assert leu_restype.which_mcsc_torsion[4] == 1
