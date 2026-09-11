import attr
import numpy
import torch

from typing import Tuple, Mapping, Union

from tmol.types import (
    NDArray,
    Tensor,
    validate_args,
)
from tmol.numeric import coord_dihedrals
from tmol.database import PatchedChemicalDatabase
from tmol.chemical import RefinedResidueType
from tmol.pose import PackedBlockTypes
from tmol.score._annotation_cache import (
    AnnotationKey,
    cached_annotation,
    store_annotation,
)

from tmol.pack.rotamer import (
    ChiSampler,
    bfs_sidechain_atoms_jit,
)

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
    duplicate_index: int = 0


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MCFingerprint:
    mc_ats: NDArray[numpy.int32][:]
    mc_at_fingerprints: Tuple[AtomFingerprint, ...]
    fingerprint: Tuple[AtomFingerprint, ...]
    at_for_fingerprint: Mapping[AtomFingerprint, int]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MCFingerprints:
    atom_mapping: Tensor[torch.int64][:, :, :, :]
    sampler_mapping: Mapping[Union[str, int], int]
    source_atom_mapping: Tensor[torch.int64][:, :]
    source_fingerprint: Tensor[torch.int64][:]


@validate_args
def create_non_sidechain_fingerprint(  # noqa: C901
    rt: RefinedResidueType,
    parents: NDArray[numpy.int32][:],
    sc_atoms: NDArray[numpy.int32][:],
    chem_db: PatchedChemicalDatabase,
):
    non_sc_atoms = numpy.nonzero(sc_atoms == 0)[0]
    # Preserve the first record for each name, matching the scalar lookups.
    element_for_type = {}
    for at in chem_db.atom_types:
        element_for_type.setdefault(at.name, at.element)
    number_for_element = {}
    for element in chem_db.element_types:
        number_for_element.setdefault(element.name, element.atomic_number)
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
        bonded_elem_name = element_for_type[bonded_atom_type]
        n_bonds[rt.bond_indices[i, 0]] += 1
        if bonded_elem_name != "H":
            n_nonh_bonds[rt.bond_indices[i, 0]] += 1
    # a connection is a bond to the neighbouring residue, so it counts toward
    #    the substituents of the atom that carries it
    for conn in rt.connection_to_idx:
        n_bonds[rt.connection_to_idx[conn]] += 1
        n_nonh_bonds[rt.connection_to_idx[conn]] += 1

    # mc_ancestors = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    # chiralities = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    non_sc_atom_fingerprints = []
    at_for_fingerprint = {}
    fp_seen_count = {}

    # Every atom is walked, but only the backbone ones are described. The
    #    duplicate index breaks ties between atoms the four fields cannot tell
    #    apart, and counting over the whole residue keeps it a property of the
    #    residue type: a sampler that calls an atom sidechain must not renumber
    #    the atoms it shares with a sampler that calls it backbone.
    is_non_sc = sc_atoms == 0
    for nsc_at in range(rt.n_atoms):
        # a protein's non-sidechain atoms all lie within one bond of the
        #    mainchain, so the branches below set this. A nucleotide's do not:
        #    its sugar is backbone and reaches two bonds out, where there is
        #    no side to be on, and the default stands.
        chirality = 0
        # find the mc atom this branches from using the kinforest. mc_anc is
        #    its position along the mainchain, which is what a fingerprint
        #    records; `atom` is that same atom's index, for reading per-atom
        #    arrays. The two coincide only when the mainchain atoms come first.
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
            if n_bonds[atom] == 4:
                # now we need to measure the chirality of the atom
                # or, rather, whether this atom is on the "left"
                # or "right" of the chiral backbone atom.
                # Measure the improper dihedral given by the
                # mc atom and two other mc atoms

                mc_ind_for_mc_anc = mc_anc
                mc1_icoor_ind, mc2_icoor_ind = _mc_inds_for_chiral_mc_atom(
                    rt, mc_atoms, mc_ind_for_mc_anc
                )

                mc_anc_icoor_ind = rt.at_to_icoor_ind[atom]

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
        atomic_number = number_for_element[element_for_type[atom_type_name]]
        base_fp = AtomFingerprint(
            mc_ind=mc_anc,
            mc_bond_dist=bonds_from_mc,
            chirality=chirality,
            element=atomic_number,
        )
        dup_idx = fp_seen_count.get(base_fp, 0)
        fp_seen_count[base_fp] = dup_idx + 1
        at_fingerprint = AtomFingerprint(
            mc_ind=mc_anc,
            mc_bond_dist=bonds_from_mc,
            chirality=chirality,
            element=atomic_number,
            duplicate_index=dup_idx,
        )

        if not is_non_sc[nsc_at]:
            continue
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
    parents, sidechain_atoms = _mainchain_region(rt, sc_roots)
    return create_non_sidechain_fingerprint(rt, parents, sidechain_atoms, chem_db)


def _mainchain_region(rt, sc_roots):
    id = rt.rotamer_kinforest.id
    parents = rt.rotamer_kinforest.parent.copy()
    parents[parents < 0] = 0
    parents[id] = id[parents]

    sc_roots = tuple(rt.atom_to_idx[at] for at in sc_roots)

    sidechain_atoms = bfs_sidechain_atoms_jit(
        parents, numpy.array(sc_roots, dtype=numpy.int32)
    )
    return parents, sidechain_atoms


def annotate_residue_type_with_sampler_fingerprints(
    restype: RefinedResidueType,
    samplers: Tuple[ChiSampler, ...],
    chem_db: PatchedChemicalDatabase,
):
    # Keep only the current task's ownership configurations. A sampler class
    # may have several instances with different roots in the same task.
    previous = list(getattr(restype, "_mc_fingerprint_annotations", {}).values())
    records, fingerprints, labels, aliases = {}, {}, {}, {}
    names = [sampler.sampler_name() for sampler in samplers]
    all_atom_fps = None
    if restype.properties.polymer.is_polymer:
        for index, sampler in enumerate(samplers):
            if not sampler.defines_rotamers_for_rt(restype):
                continue
            name = names[index]
            slot = name if names.count(name) == 1 else id(sampler)
            roots = tuple(sorted(set(sampler.first_sc_atoms_for_rt(restype))))
            key = AnnotationKey.from_sources(chem_db, settings=roots)
            cached = next(
                (
                    (old_key, value)
                    for old_key, value in previous
                    if key.matches(old_key)
                ),
                None,
            )
            if cached is None:
                # Atom descriptors do not depend on the selected roots. Compute
                # them once for this call, then keep only each retained region.
                if all_atom_fps is None:
                    all_atom_fps = create_mainchain_fingerprint(restype, (), chem_db)[1]
                _, sidechain_atoms = _mainchain_region(restype, roots)
                mc_ats = numpy.nonzero(sidechain_atoms == 0)[0]
                mc_at_fps = tuple(all_atom_fps[at] for at in mc_ats)
                at_for_fp = {all_atom_fps[at]: int(at) for at in mc_ats}
                value = MCFingerprint(
                    mc_ats=mc_ats,
                    mc_at_fingerprints=mc_at_fps,
                    fingerprint=tuple(sorted(mc_at_fps)),
                    at_for_fingerprint=at_for_fp,
                )
                cached = (key, value)
                previous.append(cached)
            records[slot] = cached
            fingerprints[slot] = cached[1]
            labels[slot] = (name, index)
            aliases[id(sampler)] = slot
    restype._mc_fingerprint_annotations = records
    restype.mc_fingerprints = fingerprints
    restype._mc_sampler_labels = labels
    restype._mc_sampler_aliases = aliases


def find_unique_fingerprints(pbt: PackedBlockTypes):
    """Copy from the union of regions retained by the current samplers.

    Source regions need not be nested. Each destination sampler still copies
    only its own retained atoms, matched against the source union by chemistry.
    """
    labels, aliases = {}, {}
    for rt in pbt.active_block_types:
        labels.update(getattr(rt, "_mc_sampler_labels", {}))
        aliases.update(getattr(rt, "_mc_sampler_aliases", {}))
    sampler_types = sorted(labels, key=labels.get)
    sampler_inds = {sampler: i for i, sampler in enumerate(sampler_types)}
    sampler_mapping = dict(sampler_inds)
    sampler_mapping.update(
        {identity: sampler_inds[slot] for identity, slot in aliases.items()}
    )
    rows = [
        [getattr(rt, "mc_fingerprints", {}).get(sampler) for sampler in sampler_types]
        for rt in pbt.active_block_types
    ]
    key = AnnotationKey.from_sources(
        *(fp for row in rows for fp in row if fp is not None),
        settings=(
            tuple(labels[sampler][0] for sampler in sampler_types),
            tuple(tuple(fp is not None for fp in row) for row in rows),
        ),
    )
    cached = cached_annotation(pbt, "_mc_packed_annotation", key)
    if cached is not None:
        # New instances with identical ownership reuse the large tensor plan.
        if cached.sampler_mapping != sampler_mapping:
            cached = attr.evolve(cached, sampler_mapping=sampler_mapping)
        pbt.mc_fingerprints = cached
        return store_annotation(pbt, "_mc_packed_annotation", key, cached)

    sources = []
    for row in rows:
        source = {}
        for fp in row:
            if fp is None:
                continue
            for atom_fp, atom in fp.at_for_fingerprint.items():
                if atom_fp in source and source[atom_fp] != atom:
                    raise ValueError(
                        "Sampler fingerprints disagree on an atom identity"
                    )
                source[atom_fp] = atom
        sources.append(source)
    source_fps = [tuple(sorted(source)) for source in sources]
    fp_sets = sorted(set(fp for fp in source_fps if fp))
    fp_to_ind = {fp: i for i, fp in enumerate(fp_sets)}
    max_n_mc_atoms = max((len(fp) for fp in fp_sets), default=0)
    source_indices = numpy.full((pbt.n_types, max_n_mc_atoms), -1, dtype=numpy.int32)
    source_fp_indices = numpy.full(pbt.n_types, -1, dtype=numpy.int32)
    for i, (source, fp) in enumerate(zip(sources, source_fps)):
        source_indices[i, : len(fp)] = [source[atom_fp] for atom_fp in fp]
        source_fp_indices[i] = fp_to_ind.get(fp, -1)

    atom_mapping = numpy.full(
        (len(sampler_types), len(fp_sets), pbt.n_types, max_n_mc_atoms),
        -1,
        dtype=numpy.int32,
    )
    for ti, row in enumerate(rows):
        for si, fingerprint in enumerate(row):
            if fingerprint is None:
                continue
            lookup = fingerprint.at_for_fingerprint
            for fi, fp in enumerate(fp_sets):
                atom_mapping[si, fi, ti, : len(fp)] = [lookup.get(at, -1) for at in fp]

    def to_device(array):
        return torch.tensor(array, dtype=torch.int64, device=pbt.device)

    fingerprints = MCFingerprints(
        atom_mapping=to_device(atom_mapping),
        sampler_mapping=sampler_mapping,
        source_atom_mapping=to_device(source_indices),
        source_fingerprint=to_device(source_fp_indices),
    )
    pbt.mc_fingerprints = fingerprints
    return store_annotation(pbt, "_mc_packed_annotation", key, fingerprints)
