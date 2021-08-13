import numpy

from tmol.chemical.restypes import ResidueTypeSet
from tmol.pack.rotamer.single_residue_kintree import (
    construct_single_residue_kintree,
    # coalesce_single_residue_kintrees,
)


def test_annotate_restypes(default_database):
    rts = ResidueTypeSet.from_database(default_database.chemical)

    for rt in rts.residue_types:
        construct_single_residue_kintree(rt)
        assert hasattr(rt, "rotamer_kintree")

        assert type(rt.rotamer_kintree.kintree_idx) == numpy.ndarray
        assert type(rt.rotamer_kintree.id) == numpy.ndarray
        assert type(rt.rotamer_kintree.doftype) == numpy.ndarray
        assert type(rt.rotamer_kintree.parent) == numpy.ndarray
        assert type(rt.rotamer_kintree.frame_x) == numpy.ndarray
        assert type(rt.rotamer_kintree.frame_y) == numpy.ndarray
        assert type(rt.rotamer_kintree.frame_z) == numpy.ndarray
        assert type(rt.rotamer_kintree.nodes) == numpy.ndarray
        assert type(rt.rotamer_kintree.scans) == numpy.ndarray
        assert type(rt.rotamer_kintree.gens) == numpy.ndarray
        assert type(rt.rotamer_kintree.n_scans_per_gen) == numpy.ndarray
        assert type(rt.rotamer_kintree.dofs_ideal) == numpy.ndarray

        assert rt.rotamer_kintree.kintree_idx.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.id.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.doftype.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.parent.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.frame_x.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.frame_y.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.frame_z.shape == (rt.n_atoms,)
        assert rt.rotamer_kintree.dofs_ideal.shape == (rt.n_atoms, 9)

        # print(rt.name)
        # print(rt.rotamer_kintree.id)
        # # print("id to kintree idx")
        # # print(rt.id_to_kintree_idx)
        # print("dofs ideal")
        # print(rt.rotamer_kintree.dofs_ideal[:, :4])
