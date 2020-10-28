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

    sc_atoms = bfs_sidechain_atoms(leu_rt, [leu_rt.atom_to_idx["CB"]])

    id = leu_rt.kintree_id
    parents = leu_rt.kintree_parent.copy()
    parents[parents < 0] = 0
    parents[id] = id[parents]

    non_sc_ats, fingerprints = create_non_sidechain_fingerprint(
        leu_rt, parents, sc_atoms, default_database.chemical
    )

    non_sc_ats_gold = numpy.array(
        sorted([leu_rt.atom_to_idx[at] for at in ("N", "CA", "C", "HA", "H", "O")]),
        dtype=numpy.int32,
    )

    numpy.testing.assert_equal(non_sc_ats_gold, non_sc_ats)

    fingerprints_gold = [
        (0, 0, 0, 7),  # N
        (1, 0, 0, 6),  # CA
        (2, 0, 0, 6),  # C
        (2, 1, 0, 8),  # O
        (0, 1, 0, 1),  # H
        (1, 1, 1, 1),  # HA
    ]
    assert fingerprints == fingerprints_gold


def test_create_non_sc_fingerprint_smoke(default_database):
    torch_device = torch.device("cpu")
    rts = ResidueTypeSet.from_database(default_database.chemical)

    canonical_aas = [
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "ILE",
        "LYS",
        "LEU",
        "MET",
        "ASN",
        "PRO",
        "GLN",
        "ARG",
        "SER",
        "THR",
        "VAL",
        "TRP",
        "TYR",
    ]

    for aa in canonical_aas:
        rt = rts.restype_map[aa][0]

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
