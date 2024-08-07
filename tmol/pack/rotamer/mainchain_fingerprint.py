import attr
import numpy
import torch

from typing import Tuple, Mapping

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.numeric.dihedrals import coord_dihedrals
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes

from tmol.pack.rotamer.chi_sampler import ChiSampler
from tmol.pack.rotamer.bfs_sidechain import bfs_sidechain_atoms_jit


# what atoms should we copy over?
# everything north of "first sidechain atom"?
# let's have a map from rt x bb-type --> atom-indices on that rt for those bb
# and then when we want to map between two rts, we ask "what is their rt compatibility"?
# and then use that mapping

# so
# all canonical aas except proline are class 1
# pro is class 2
# gly is class 3
#
# class 1 has n, ca, c, o, h, and ha
# class 2 has n, ca, c, o, and ha
# class 3 has n, ca, c, o, and the "left" ha

# how do we tell what classes of backbones there are?
# we ask:
# what atoms are upstream of the first sidechain atom
# for each atom that's upstream of the first sidechain atom
# who is chemically bound to it, what is the chirality
# of that connection, and what is the element type of that
# connection

# then we need to hash that
# (how???)
# atoms then should be sorted along mainchain?? and then
# with chirality

# n -- > (0, 0, 0, 7)
# h -- > (0, 1, 0, 1)
# ca --> (1, 0, 0, 6)
# ha --> (1, 1, 1, 1)
# c  --> (2, 0, 0, 6)
# o  --> (2, 1, 0, 8)

# position 0: position along the backbone or backbone you're bonded to
# position 1: number of bonds from the backbone
# position 2: chirality: 0 - achiral, 1 - left, 2 - right
# position 3: element

# how do I determine chirality?
#
# if bb atom has three chemical bonds, then
# treat it as achiral.
# if it has four chemical bonds, then
# measure chirality of 4th bond by taking
# the dot product of sc-i and the cross
# product of (p_i - p_{i-1}) and (p_{i+1}, p_i)
# if it's positive, then chirality value of 1
# if it's negative, then chirality value of 2

# and then the 4th column is the element, so, that needs to be encoded somehow...

# how do we sort atoms further from the backbone?
# what about when something like: put into the chirality position
# a counter so that things further from the backbone get noted
# with a higher count; how can you guarantee uniqueness, though??
# maybe it should be like an array with an offset based on the chirality
# of its ancestors back to the backbone where you put


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomFingerprint:
    mc_ind: int
    mc_bond_dist: int
    chirality: int
    element: int


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MCFingerprint:
    mc_ats: NDArray[numpy.int32][:]
    mc_at_fingerprints: Tuple[AtomFingerprint, ...]
    fingerprint: Tuple[AtomFingerprint, ...]
    at_for_fingerprint: Mapping[AtomFingerprint, int]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MCFingerprints:
    atom_mapping: Tensor[torch.int32][:, :, :, :]  # make int64
    sampler_mapping: Mapping[str, int]
    max_sampler: Tensor[torch.int32][:]
    max_fingerprint: Tensor[torch.int32][:]


@validate_args
def create_non_sidechain_fingerprint(
    rt: RefinedResidueType,
    parents: NDArray[numpy.int32][:],
    sc_atoms: NDArray[numpy.int32][:],
    chem_db: PatchedChemicalDatabase,
):
    non_sc_atoms = numpy.nonzero(sc_atoms == 0)[0]
    # TO DO: mainchain_atoms determined programatically from
    # shortest path between up- and down-connection atoms
    mc_at_names = rt.properties.polymer.mainchain_atoms
    mc_atoms = numpy.array(
        [rt.atom_to_idx[at] for at in mc_at_names], dtype=numpy.int32
    )
    # mc_ind: which mainchain atom [0..n_mc_atoms) is a particular
    # atom by atom index
    mc_ind = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    mc_ind[mc_atoms] = numpy.arange(mc_atoms.shape[0], dtype=numpy.int32)

    # fd temporary fix for terminal variants
    # count the number of bonds to non-H for each atom
    # apl maybe undoing this change
    n_bonds = numpy.zeros(rt.n_atoms, dtype=numpy.int32)
    n_nonh_bonds = numpy.zeros(rt.n_atoms, dtype=numpy.int32)
    for i in range(rt.bond_indices.shape[0]):
        bonded_atom_type = rt.atoms[rt.bond_indices[i, 1]].atom_type
        bonded_elem_name = next(
            at.element for at in chem_db.atom_types if at.name == bonded_atom_type
        )
        n_bonds[rt.bond_indices[i, 0]] += 1
        if bonded_elem_name != "H":
            n_nonh_bonds[rt.bond_indices[i, 0]] += 1
    for conn in rt.connection_to_idx:
        n_bonds[rt.bond_indices[i, 0]] += 1
        n_nonh_bonds[rt.connection_to_idx[conn]] += 1

    # mc_ancestors = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    # chiralities = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    non_sc_atom_fingerprints = []
    at_for_fingerprint = {}

    for nsc_at in non_sc_atoms:
        # find the index of the mc atom this branches from using the kinforest
        mc_anc = mc_ind[nsc_at]
        bonds_from_mc = 0
        atom = nsc_at
        for _ in range(rt.n_atoms):
            if mc_anc != -1:
                break
            par = parents[atom]
            mc_anc = mc_ind[par]
            atom = par
            bonds_from_mc += 1

        # now lets figure out the chirality of this atom
        # "chirality" here is interpretted in the most liberal of ways
        # where it can refer to L or D for H-alpha (connected to CA), or
        # achiral for the H or O atoms bound to the planar N and C atoms,
        # but it will also interpret something as "chiral" if its MC atom
        # has four substituents even when two of those substituents are
        # chemically identical; i.e. as long as the atoms for thse
        # substituents have different names they are different, thus the
        # atom that binds them is chiral. So, unintuitively, glycine's CA
        # will be declared as chiral and we will calculate the chirality
        # of its substituents

        if bonds_from_mc == 0:
            chirality = 0
        elif bonds_from_mc == 1:
            if n_bonds[mc_anc] == 4:
                # now we need to measure the chirality of the atom
                # or, rather, whether this atom is on the "left"
                # or "right" of the chiral backbone atom.
                # Measure the improper dihedral given by the
                # mc atom and two other mc atoms

                mc_ind_for_mc_anc = mc_ind[mc_anc]
                mc1_icoor_ind, mc2_icoor_ind = _mc_inds_for_chiral_mc_atom(
                    rt, mc_atoms, mc_ind_for_mc_anc
                )

                mc_anc_icoor_ind = rt.at_to_icoor_ind[mc_anc]

                def t64(coord):
                    return torch.tensor(coord, dtype=torch.float64).unsqueeze(0)

                at1_coord = t64(rt.ideal_coords[mc1_icoor_ind])
                at2_coord = t64(rt.ideal_coords[mc_anc_icoor_ind])
                at3_coord = t64(rt.ideal_coords[mc2_icoor_ind])
                at4_coord = t64(rt.ideal_coords[rt.at_to_icoor_ind[nsc_at]])

                # now we have four coordinates, measure the dihedral
                dihe = numpy.degrees(
                    coord_dihedrals(at4_coord, at2_coord, at1_coord, at3_coord).numpy()[
                        0
                    ]
                )
                # some atoms are going to be placed in the plane
                # defined by the three "main chain" atoms. If the
                # atoms are within an epsilon of a dihedral angle of 0 or 180
                # then we will label their "chirality" as 3
                epsilon = 1  # 1 degree of fudge for planarity
                abs_dihe = numpy.absolute(dihe)
                if abs_dihe < epsilon or numpy.absolute(180 - abs_dihe) < epsilon:
                    chirality = 3
                elif dihe > 0:
                    chirality = 1
                else:
                    chirality = 2
            else:
                chirality = 0

        atom_type_name = rt.atoms[nsc_at].atom_type
        elem_name = next(
            at.element for at in chem_db.atom_types if at.name == atom_type_name
        )
        atomic_number = next(
            el.atomic_number for el in chem_db.element_types if el.name == elem_name
        )
        at_fingerprint = AtomFingerprint(
            mc_ind=mc_anc,
            mc_bond_dist=bonds_from_mc,
            chirality=chirality,
            element=atomic_number,
        )

        non_sc_atom_fingerprints.append(at_fingerprint)
        at_for_fingerprint[at_fingerprint] = nsc_at
    return non_sc_atoms, tuple(non_sc_atom_fingerprints), at_for_fingerprint


def _mc_inds_for_chiral_mc_atom(
    rt,
    mc_atoms,
    mc_ind_for_mc_anc,
):
    if mc_ind_for_mc_anc == 0:
        if "down" in rt.icoors:
            mc1_icoor_ind = rt.icoors_index["down"]
            mc2_icoor_ind = rt.at_to_icoor_ind[mc_atoms[mc_ind_for_mc_anc + 1]]
        else:
            # ok, so, this gets a little complicated if
            # there are fewer than 3 mainchain atoms
            # so let's handle those cases later
            if len(mc_atoms) >= 3:
                mc1_icoor_ind = rt.at_to_icoor_ind[mc_atoms[mc_ind_for_mc_anc + 1]]
                mc2_icoor_ind = rt.at_to_icoor_ind[mc_atoms[mc_ind_for_mc_anc + 2]]
            elif len(mc_atoms) == 2:
                # TO DO
                # ?? I don't know
                # I am having trouble envisioning this "main chain"
                raise NotImplementedError(
                    "No logic yet to handle packing a two-atom main chain"
                )
            else:
                # TO DO
                # ie. elif len(mc_atoms) == 1:
                # ?? Perhaps this might come up if the
                # block type is just a single hydroxyl that's been
                # sheared off a sugar residue to be packed
                # independently and the "sidechain" is the hydroxyl H
                # and the "mainchain" is the hydroxyl O.
                raise NotImplementedError(
                    "No logic yet to handle packing a one-atom main chain"
                )

    elif mc_ind_for_mc_anc == len(mc_atoms) - 1:
        if "up" in rt.icoors:
            mc1_icoor_ind = rt.at_to_icoor_ind[mc_atoms[mc_ind_for_mc_anc - 1]]
            mc2_icoor_ind = rt.icoors_index["up"]
        else:
            # ok, so this gets a little complicated if there are
            # fewer than 3 mainchain atoms
            if len(mc_atoms) == 3:
                mc1_icoor_ind = rt.at_to_icoor_ind[mc_atoms[mc_ind_for_mc_anc - 1]]
                mc2_icoor_ind = rt.at_to_icoor_ind[mc_atoms[mc_ind_for_mc_anc - 2]]
            else:
                # TO DO
                # i.e. len(mc_atoms) == 2; if there is only one MC atom, then
                # it will be the first MC atom, and then we will not reach
                # the "elif mc_ind_for_mc_anc == len(mc_atoms) - 1"
                # I don't know what this mainchain looks like
                raise NotImplementedError(
                    "No logic yet to handle packing a two-atom main chain"
                )
    else:
        # somewhere in the middle of the main chain (e.g. CA)
        mc1_icoor_ind = rt.at_to_icoor_ind[mc_atoms[mc_ind_for_mc_anc - 1]]
        mc2_icoor_ind = rt.at_to_icoor_ind[mc_atoms[mc_ind_for_mc_anc + 1]]

    return mc1_icoor_ind, mc2_icoor_ind


@validate_args
def create_mainchain_fingerprint(
    rt: RefinedResidueType, sc_roots: Tuple[str, ...], chem_db: PatchedChemicalDatabase
):
    id = rt.rotamer_kinforest.id
    parents = rt.rotamer_kinforest.parent.copy()
    parents[parents < 0] = 0
    parents[id] = id[parents]

    sc_roots = tuple(rt.atom_to_idx[at] for at in sc_roots)

    sidechain_atoms = bfs_sidechain_atoms_jit(
        parents, numpy.array(sc_roots, dtype=numpy.int32)
    )
    return create_non_sidechain_fingerprint(rt, parents, sidechain_atoms, chem_db)


def annotate_residue_type_with_sampler_fingerprints(
    restype: RefinedResidueType,
    samplers: Tuple[ChiSampler, ...],
    chem_db: PatchedChemicalDatabase,
):
    for sampler in samplers:
        if sampler.defines_rotamers_for_rt(restype):
            if hasattr(restype, "mc_fingerprints"):
                if sampler.sampler_name() in restype.mc_fingerprints:
                    continue
            else:
                setattr(restype, "mc_fingerprints", {})

            sc_roots = sampler.first_sc_atoms_for_rt(restype)
            mc_ats, mc_at_fps, at_for_fp = create_mainchain_fingerprint(
                restype, sc_roots, chem_db
            )
            fingerprint = tuple(sorted(mc_at_fps))
            restype.mc_fingerprints[sampler.sampler_name()] = MCFingerprint(
                mc_ats=mc_ats,
                mc_at_fingerprints=mc_at_fps,
                fingerprint=fingerprint,
                at_for_fingerprint=at_for_fp,
            )


def find_max_length_fp_among_res_samplers(
    pbt: PackedBlockTypes, sampler_types, fp_sets
):
    fp_to_ind = {fp: i for i, fp in enumerate(fp_sets)}
    n_fps_for_rt = numpy.zeros((pbt.n_types,), dtype=numpy.int32)
    max_sampler_for_rt = numpy.full((pbt.n_types,), -1, dtype=numpy.int32)
    max_fp_for_rt = numpy.full((pbt.n_types,), -1, dtype=numpy.int32)

    for i, rt in enumerate(pbt.active_block_types):
        if hasattr(rt, "mc_fingerprints"):
            for j, sampler in enumerate(sampler_types):
                if sampler in rt.mc_fingerprints:
                    j_len = len(rt.mc_fingerprints[sampler].mc_ats)
                    if j_len > n_fps_for_rt[i]:
                        n_fps_for_rt[i] = j_len
                        max_sampler_for_rt[i] = j
                        max_fp_for_rt[i] = fp_to_ind[
                            rt.mc_fingerprints[sampler].fingerprint
                        ]
    return (n_fps_for_rt, max_sampler_for_rt, max_fp_for_rt, fp_to_ind)


def find_unique_fingerprints(
    pbt: PackedBlockTypes,
):
    sampler_types = set()
    for rt in pbt.active_block_types:
        if hasattr(rt, "mc_fingerprints"):
            for sampler in rt.mc_fingerprints:
                sampler_types.add(sampler)

    # we do not need to re-annotate this PackedBlockTypes object if there
    # are no sidechain samplers that it has not encountered before
    if hasattr(pbt, "mc_atom_mapping"):
        all_found = True
        for bt in sampler_types:
            if bt not in pbt.mc_atom_mapping:
                all_found = False
                break
        if all_found:
            return
    sampler_types = sorted(list(sampler_types))
    n_samplers = len(sampler_types)
    sampler_inds = {sampler: i for i, sampler in enumerate(sampler_types)}

    fp_sets = set()
    for rt in pbt.active_block_types:
        if hasattr(rt, "mc_fingerprints"):
            for _, mcfps in rt.mc_fingerprints.items():
                fp_sets.add(mcfps.fingerprint)
    fp_sets = sorted(fp_sets)

    n_mcs = len(fp_sets)
    max_n_mc_atoms = max(len(fp) for fp in fp_sets)

    (
        n_fps_for_rt,
        max_sampler_for_rt,
        max_fp_for_rt,
        fp_to_ind,
    ) = find_max_length_fp_among_res_samplers(pbt, sampler_types, fp_sets)

    # ok, we need have n mainchain types
    # and we have m residue types
    # we have n x n different ways to map atoms from one mainchain
    # type onto another mainchain type

    # we will create an b x n x m x n-ats array that says which atom l
    # on mc-type j maps to which atom on residue type k as defined by
    # sampler i

    mc_atom_inds_for_rt_for_sampler = numpy.full(
        (n_samplers, n_mcs, pbt.n_types, max_n_mc_atoms), -1, dtype=numpy.int32
    )
    for ii, sampler in enumerate(sampler_types):
        for jj, fp in enumerate(fp_sets):
            for kk, rt in enumerate(pbt.active_block_types):
                # now we'er going to find the index of the mainchain atom
                if not hasattr(rt, "mc_fingerprints"):
                    continue
                if sampler not in rt.mc_fingerprints:
                    continue
                rt_fingerprint = rt.mc_fingerprints[sampler]
                for ll, at_fp in enumerate(fp):
                    mc_atom_inds_for_rt_for_sampler[ii, jj, kk, ll] = (
                        rt_fingerprint.at_for_fingerprint.get(at_fp, -1)
                    )

    def _t(arr):
        return torch.tensor(arr, dtype=torch.int64, device=pbt.device)

    fingerprints = MCFingerprints(
        atom_mapping=_t(mc_atom_inds_for_rt_for_sampler),
        sampler_mapping=sampler_inds,
        max_sampler=_t(max_sampler_for_rt),
        max_fingerprint=_t(max_fp_for_rt),
    )
    setattr(pbt, "mc_fingerprints", fingerprints)
