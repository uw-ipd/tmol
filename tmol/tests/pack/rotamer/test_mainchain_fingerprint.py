import numpy
import torch

from tmol.system.restypes import RefinedResidueType, ResidueTypeSet

# from tmol.pack.rotamer.build_rotamers import annotate_restype

from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.pack.rotamer.single_residue_kintree import construct_single_residue_kintree
from tmol.pack.rotamer.mainchain_fingerprint import (
    create_non_sidechain_fingerprint,
    create_mainchain_fingerprint,
    AtomFingerprint,
    annotate_residue_type_with_sampler_fingerprints,
)
from tmol.pack.rotamer.bfs_sidechain import bfs_sidechain_atoms
from tmol.pack.rotamer.dunbrack_chi_sampler import DunbrackChiSampler


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

    fingerprints_gold = (
        AtomFingerprint(0, 0, 0, 7),  # N
        AtomFingerprint(1, 0, 0, 6),  # CA
        AtomFingerprint(2, 0, 0, 6),  # C
        AtomFingerprint(2, 1, 0, 8),  # O
        AtomFingerprint(0, 1, 0, 1),  # H
        AtomFingerprint(1, 1, 1, 1),  # HA
    )
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

        non_sc_ats, fingerprints = create_mainchain_fingerprint(
            rt, (sc_at_root,), default_database.chemical
        )


def test_annotate_rt_w_mainchain_fingerprint(default_database):
    torch_device = torch.device("cpu")
    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_rt = rts.restype_map["LEU"][0]
    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)

    construct_single_residue_kintree(leu_rt)
    annotate_residue_type_with_sampler_fingerprints(
        leu_rt, [dun_sampler], default_database.chemical
    )

    assert hasattr(leu_rt, "mc_fingerprints")
    assert dun_sampler.sampler_name() in leu_rt.mc_fingerprints

    non_sc_ats_gold = numpy.array(
        sorted([leu_rt.atom_to_idx[at] for at in ("N", "CA", "C", "HA", "H", "O")]),
        dtype=numpy.int32,
    )

    mc_fingerprint = leu_rt.mc_fingerprints[dun_sampler.sampler_name()]

    numpy.testing.assert_equal(non_sc_ats_gold, mc_fingerprint.mc_ats)

    fingerprints_gold = (
        AtomFingerprint(0, 0, 0, 7),  # N
        AtomFingerprint(1, 0, 0, 6),  # CA
        AtomFingerprint(2, 0, 0, 6),  # C
        AtomFingerprint(2, 1, 0, 8),  # O
        AtomFingerprint(0, 1, 0, 1),  # H
        AtomFingerprint(1, 1, 1, 1),  # HA
    )
    assert mc_fingerprint.mc_at_fingerprints == fingerprints_gold
