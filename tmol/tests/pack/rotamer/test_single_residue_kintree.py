import numpy
import torch

from tmol.system.restypes import RefinedResidueType, ResidueTypeSet
from tmol.pack.rotamer.build_rotamers import annotate_restype
from tmol.pack.rotamer.single_residue_kintree import (
    construct_single_residue_kintree,
    coalesce_single_residue_kintrees,
)


def test_annotate_restypes(default_database):
    rts = ResidueTypeSet.from_database(default_database.chemical)

    for rt in rts.residue_types:
        annotate_restype(rt)
        assert hasattr(rt, "kintree_id")
        assert hasattr(rt, "kintree_doftype")
        assert hasattr(rt, "kintree_parent")
        assert hasattr(rt, "kintree_frame_x")
        assert hasattr(rt, "kintree_frame_y")
        assert hasattr(rt, "kintree_frame_z")
        assert hasattr(rt, "kintree_nodes")
        assert hasattr(rt, "kintree_scans")
        assert hasattr(rt, "kintree_gens")
        assert hasattr(rt, "kintree_n_scans_per_gen")

        assert type(rt.kintree_id) == numpy.ndarray
        assert type(rt.kintree_doftype) == numpy.ndarray
        assert type(rt.kintree_parent) == numpy.ndarray
        assert type(rt.kintree_frame_x) == numpy.ndarray
        assert type(rt.kintree_frame_y) == numpy.ndarray
        assert type(rt.kintree_frame_z) == numpy.ndarray

        assert rt.kintree_id.shape == (rt.n_atoms,)
        assert rt.kintree_doftype.shape == (rt.n_atoms,)
        assert rt.kintree_parent.shape == (rt.n_atoms,)
        assert rt.kintree_frame_x.shape == (rt.n_atoms,)
        assert rt.kintree_frame_y.shape == (rt.n_atoms,)
        assert rt.kintree_frame_z.shape == (rt.n_atoms,)
