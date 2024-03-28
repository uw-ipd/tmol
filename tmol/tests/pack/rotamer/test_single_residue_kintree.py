import numpy

from tmol.chemical.restypes import ResidueTypeSet
from tmol.pack.rotamer.single_residue_kinforest import (
    construct_single_residue_kinforest,
    # coalesce_single_residue_kinforests,
)


def test_annotate_restypes(default_database):
    rts = ResidueTypeSet.from_database(default_database.chemical)

    for rt in rts.residue_types:
        construct_single_residue_kinforest(rt)
        assert hasattr(rt, "rotamer_kinforest")

        assert isinstance(rt.rotamer_kinforest.kinforest_idx, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.id, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.doftype, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.parent, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.frame_x, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.frame_y, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.frame_z, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.nodes, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.scans, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.gens, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.n_scans_per_gen, numpy.ndarray)
        assert isinstance(rt.rotamer_kinforest.dofs_ideal, numpy.ndarray)

        assert rt.rotamer_kinforest.kinforest_idx.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.id.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.doftype.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.parent.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.frame_x.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.frame_y.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.frame_z.shape == (rt.n_atoms,)
        assert rt.rotamer_kinforest.dofs_ideal.shape == (rt.n_atoms, 9)

        # print(rt.name)
        # print(rt.rotamer_kinforest.id)
        # # print("id to kinforest idx")
        # # print(rt.id_to_kinforest_idx)
        # print("dofs ideal")
        # print(rt.rotamer_kinforest.dofs_ideal[:, :4])
