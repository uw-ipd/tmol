import numpy
import torch

from tmol.chemical.restypes import ResidueTypeSet

from tmol.score.dunbrack.params import DunbrackParamResolver
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pack.rotamer.single_residue_kintree import construct_single_residue_kintree
from tmol.pack.rotamer.mainchain_fingerprint import (
    create_non_sidechain_fingerprint,
    create_mainchain_fingerprint,
    AtomFingerprint,
    annotate_residue_type_with_sampler_fingerprints,
    find_unique_fingerprints,
)
from tmol.pack.rotamer.bfs_sidechain import bfs_sidechain_atoms
from tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler import DunbrackChiSampler
from tmol.pack.rotamer.fixed_aa_chi_sampler import FixedAAChiSampler


def test_create_non_sidechain_fingerprint(default_database):
    rts = ResidueTypeSet.from_database(default_database.chemical)
    leu_rt = rts.restype_map["LEU"][0]
    construct_single_residue_kintree(leu_rt)

    sc_atoms = bfs_sidechain_atoms(leu_rt, [leu_rt.atom_to_idx["CB"]])

    id = leu_rt.rotamer_kintree.id
    parents = leu_rt.rotamer_kintree.parent.copy()
    parents[parents < 0] = 0
    parents[id] = id[parents]

    non_sc_ats, fingerprints, _ = create_non_sidechain_fingerprint(
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

        create_mainchain_fingerprint(rt, (sc_at_root,), default_database.chemical)


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

    atom_fingerprints_gold = (
        AtomFingerprint(0, 0, 0, 7),  # N
        AtomFingerprint(1, 0, 0, 6),  # CA
        AtomFingerprint(2, 0, 0, 6),  # C
        AtomFingerprint(2, 1, 0, 8),  # O
        AtomFingerprint(0, 1, 0, 1),  # H
        AtomFingerprint(1, 1, 1, 1),  # HA
    )
    assert mc_fingerprint.mc_at_fingerprints == atom_fingerprints_gold

    fingerprint_gold = (
        AtomFingerprint(0, 0, 0, 7),  # N
        AtomFingerprint(0, 1, 0, 1),  # H
        AtomFingerprint(1, 0, 0, 6),  # CA
        AtomFingerprint(1, 1, 1, 1),  # HA
        AtomFingerprint(2, 0, 0, 6),  # C
        AtomFingerprint(2, 1, 0, 8),  # O
    )

    assert mc_fingerprint.fingerprint == fingerprint_gold


def test_merge_fingerprints(default_database):
    torch_device = torch.device("cpu")
    rts = ResidueTypeSet.from_database(default_database.chemical)

    param_resolver = DunbrackParamResolver.from_database(
        default_database.scoring.dun, torch_device
    )
    dun_sampler = DunbrackChiSampler.from_database(param_resolver)
    fixed_sampler = FixedAAChiSampler()

    canonical_aas = [
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
        "GLY",
        "HIS",
        "HIS_D",
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

    rt_list = [
        next(rt for rt in rts.residue_types if rt.name == aa) for aa in canonical_aas
    ]

    for rt in rt_list:
        construct_single_residue_kintree(rt)
        annotate_residue_type_with_sampler_fingerprints(
            rt, [dun_sampler, fixed_sampler], default_database.chemical
        )

    pbt = PackedBlockTypes.from_restype_list(rt_list, device=torch_device)
    find_unique_fingerprints(pbt)

    # we should find that the pbt has been annotated
    # we should find that it has discovered two backbone types
    # we should see that, when mapping proline onto a leucine,
    # that proline has a -1 for the HA atom of leucine's mainchain.
    # we should see that otherwise the standard set of atoms
    # map to each other, except for glycine which uses 2HA to map
    # to HA.

    assert hasattr(pbt, "mc_fingerprints")

    standard_mc_atoms = ["N", "H", "CA", "HA", "C", "O"]
    glycine_mc_atoms = ["N", "H", "CA", "1HA", "C", "O"]

    standard_mc_atoms_w_pro = ["N", "CA", "HA", "C", "O"]
    glycine_mc_atoms_w_pro = ["N", "CA", "1HA", "C", "O"]

    def which_atoms(orig_rt, target_rt):
        if orig_rt.name == "PRO":
            if target_rt.name == "GLY":
                return glycine_mc_atoms_w_pro
            else:
                return standard_mc_atoms_w_pro
        else:
            if target_rt.name == "GLY":
                return glycine_mc_atoms
            else:
                return standard_mc_atoms

    assert dun_sampler.sampler_name() in pbt.mc_fingerprints.sampler_mapping
    dun_sampler_ind = pbt.mc_fingerprints.sampler_mapping[dun_sampler.sampler_name()]

    assert fixed_sampler.sampler_name() in pbt.mc_fingerprints.sampler_mapping
    fixed_sampler_ind = pbt.mc_fingerprints.sampler_mapping[
        fixed_sampler.sampler_name()
    ]

    # pro_pbt_ind = pbt.restype_index.get_indexer(["PRO"])[0]
    # gly_pbt_ind = pbt.restype_index.get_indexer(["GLY"])[0]
    # leu_pbt_ind = pbt.restype_index.get_indexer(["LEU"])[0]
    #
    # assert pro_pbt_ind == dun_sampler_ind
    # assert gly_pbt_ind == fixed_sampler_ind
    # assert leu_pbt_ind == dun_sampler_ind

    # print(pro_pbt_ind, gly_pbt_ind, leu_pbt_ind)

    assert pbt.mc_fingerprints.atom_mapping.shape == (2, 2, 21, 6)

    for i, rt_orig in enumerate(pbt.active_block_types):
        orig_rt_sampler = pbt.mc_fingerprints.max_sampler[i]
        orig_max_fp = pbt.mc_fingerprints.max_fingerprint[i]
        orig_mc_ats = which_atoms(rt_orig, rt_orig)

        for j, rt_new in enumerate(pbt.active_block_types):
            new_mc_ats = which_atoms(rt_orig, rt_new)
            if fixed_sampler.defines_rotamers_for_rt(rt_new):
                new_rt_sampler = fixed_sampler_ind
            else:
                new_rt_sampler = dun_sampler_ind

            # now the atom mapping:
            for k in range(6):
                k_orig = pbt.mc_fingerprints.atom_mapping[
                    orig_rt_sampler, orig_max_fp, i, k
                ]
                k_new = pbt.mc_fingerprints.atom_mapping[
                    new_rt_sampler, orig_max_fp, j, k
                ]

                if k_orig >= 0 and k_new >= 0:
                    assert rt_orig.atoms[k_orig].name == orig_mc_ats[k]
                    assert rt_new.atoms[k_new].name == new_mc_ats[k]
                elif k_orig >= 0 and k_new == -1:
                    assert new_mc_ats[k] not in rt_new.atom_to_idx
                elif k_orig == -1 and k_new == -1:
                    # there are simply fewer mc atoms for this fingerprint
                    # and we have come to the end of the list
                    assert True
                else:
                    # this should never happen
                    assert k_orig >= 0
