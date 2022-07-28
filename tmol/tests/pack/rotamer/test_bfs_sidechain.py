import numpy
import torch

from tmol.chemical.restypes import ResidueTypeSet
from tmol.pack.rotamer.bfs_sidechain import bfs_sidechain_atoms
from tmol.pack.rotamer.single_residue_kinforest import (
    construct_single_residue_kinforest
)


def test_identify_sidechain_atoms_from_roots(default_database):
    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_rt = rts.restype_map["LEU"][0]

    construct_single_residue_kinforest(leu_rt)

    sc_ats = bfs_sidechain_atoms(leu_rt, [leu_rt.atom_to_idx["CB"]])
    atom_names = numpy.array([at.name for at in leu_rt.atoms], dtype=str)
    sc_ats = set(atom_names[sc_ats != 0])
    gold_sc_ats = [
        "CB",
        "1HB",
        "2HB",
        "CG",
        "HG",
        "CD1",
        "1HD1",
        "2HD1",
        "3HD1",
        "CD2",
        "1HD2",
        "2HD2",
        "3HD2",
    ]
    assert len(gold_sc_ats) == len(sc_ats)
    for gold_at in gold_sc_ats:
        assert gold_at in sc_ats
