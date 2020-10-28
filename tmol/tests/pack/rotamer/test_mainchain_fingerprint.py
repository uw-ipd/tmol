import numpy
import torch

from tmol.system.restypes import RefinedResidueType, ResidueTypeSet

# from tmol.pack.rotamer.build_rotamers import annotate_restype
from tmol.pack.rotamer.single_residue_kintree import construct_single_residue_kintree
from tmol.pack.rotamer.mainchain_fingerprint import create_non_sidechain_fingerprint
from tmol.pack.rotamer.bfs_sidechain import bfs_sidechain_atoms


def test_create_non_sidechain_fingerprint(default_database):
    torch_device = torch.device("cpu")
    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_rt = rts.restype_map["LEU"][0]
    construct_single_residue_kintree(leu_rt)
    print(leu_rt.kintree_parent)

    sc_atoms = bfs_sidechain_atoms(leu_rt, [leu_rt.atom_to_idx["CB"]])

    id = leu_rt.kintree_id
    parents = leu_rt.kintree_parent.copy()
    parents[parents < 0] = 0
    parents[id] = id[parents]

    non_sc_ats, fingerprints = create_non_sidechain_fingerprint(
        leu_rt, parents, sc_atoms, default_database.chemical
    )
    print("fingerprints")
    print(fingerprints)


def test_create_non_sc_fingerprint2(default_database):
    torch_device = torch.device("cpu")
    rts = ResidueTypeSet.from_database(default_database.chemical)

    for rt in rts.residue_types:
        if rt.name == "HOH":
            continue
        construct_single_residue_kintree(rt)

        sc_at_root = "CB" if rt.name != "GLY" else "2HA"
        sc_atoms = bfs_sidechain_atoms(rt, [rt.atom_to_idx[sc_at_root]])

        id = rt.kintree_id
        parents = rt.kintree_parent.copy()
        parents[parents < 0] = 0
        parents[id] = id[parents]

        non_sc_ats, fingerprints = create_non_sidechain_fingerprint(
            rt, parents, sc_atoms, default_database.chemical
        )
        print("fingerprints", rt.name)
        print(fingerprints)
