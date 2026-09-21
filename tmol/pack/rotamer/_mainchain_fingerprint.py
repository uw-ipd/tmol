"""Match retained backbone atoms across residue types and sampler ownership."""

import attr
import numpy
import torch

from typing import Tuple, Mapping, Union

from tmol.types import (
    NDArray,
    Tensor,
    validate_args,
)
from tmol.numeric._dihedrals import _numpy_coord_dihedrals
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


@attr.s(auto_attribs=True, frozen=True, slots=True)
class AtomFingerprint:
    """Backbone position, tree distance, side of its frame and atomic number.

    Chirality is 0 when unclassified, 1/2 on opposite sides, and 3 in-plane.
    This distinguishes even glycine's equivalent hydrogens. Duplicate indices
    disambiguate atoms with the same descriptor independently of sampler roots.
    """

    mc_ind: int
    mc_bond_dist: int
    chirality: int
    element: int
    duplicate_index: int = 0


@attr.s(auto_attribs=True, frozen=True, slots=True)
class MCFingerprint:
    """Main-chain atom fingerprint and its residue-local atom mapping."""

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
    mc_at_names = rt.properties.polymer.mainchain_atoms
    mc_atoms = numpy.array(
        [rt.atom_to_idx[at] for at in mc_at_names], dtype=numpy.int32
    )
    # mc_ind: which mainchain atom [0..n_mc_atoms) is a particular
    # atom by atom index
    mc_ind = numpy.full(rt.n_atoms, -1, dtype=numpy.int32)
    mc_ind[mc_atoms] = numpy.arange(mc_atoms.shape[0], dtype=numpy.int32)

    # A kinematic parent tree has to cut cycles. For polymer rings branching
    # from the mainchain (notably the nucleotide sugar), following parents can
    # therefore reach the tree root without ever visiting a declared mainchain
    # atom. Keep the parent-tree path as the normal definition, but fall back to
    # the shortest chemical-bond path for those atoms.
    bonded_neighbors = [[] for _ in range(rt.n_atoms)]
    for atom1, atom2 in rt.bond_indices:
        bonded_neighbors[atom1].append(atom2)
        bonded_neighbors[atom2].append(atom1)

    def closest_mainchain_atom(atom):
        visited = {atom}
        frontier = [atom]
        for bond_distance in range(rt.n_atoms):
            mainchain = sorted(a for a in frontier if mc_ind[a] >= 0)
            if mainchain:
                return mainchain[0], bond_distance
            frontier = sorted(
                {
                    neighbor
                    for current in frontier
                    for neighbor in bonded_neighbors[current]
                    if neighbor not in visited
                }
            )
            visited.update(frontier)
        raise ValueError(f"Atom {atom} in {rt.name} is disconnected from its mainchain")

    # Each directed internal edge and each external connection is one neighbour.
    n_bonds = numpy.bincount(
        numpy.concatenate((rt.bond_indices[:, 0], rt.ordered_connection_atoms)),
        minlength=rt.n_atoms,
    )
    icoor_coords = rt.ideal_coords
    chiral_frames = {}

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
            if par < 0 or par == atom:
                break
            mc_anc = mc_ind[par]
            atom = par
            bonds_from_mc += 1

        if mc_anc == -1:
            atom, bonds_from_mc = closest_mainchain_atom(nsc_at)
            mc_anc = mc_ind[atom]

        # Four substituents need side-of-plane labels even if chemically equal.
        if bonds_from_mc == 1 and n_bonds[atom] == 4:
            if mc_anc not in chiral_frames:
                chiral_frames[mc_anc] = _mc_inds_for_chiral_mc_atom(
                    rt, mc_atoms, mc_anc
                )
            first, second = chiral_frames[mc_anc]
            dihe = numpy.degrees(
                _numpy_coord_dihedrals(
                    icoor_coords[rt.at_to_icoor_ind[nsc_at]],
                    icoor_coords[rt.at_to_icoor_ind[atom]],
                    icoor_coords[first],
                    icoor_coords[second],
                )
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


def _mc_inds_for_chiral_mc_atom(rt, mc_atoms, mc_index):
    """Two reference points defining the sides of a mainchain atom.

    Extend the backbone with its connection coordinates. Short caps lacking a
    third backbone point use their declared construction frame. These labels
    distinguish substituent positions; they are not CIP stereochemical labels.
    """
    backbone = [int(rt.at_to_icoor_ind[atom]) for atom in mc_atoms]
    if "down" in rt.icoors_index:
        backbone.insert(0, rt.icoors_index["down"])
        mc_index += 1
    if "up" in rt.icoors_index:
        backbone.append(rt.icoors_index["up"])
    if len(backbone) >= 3:
        if mc_index == 0:
            return backbone[1], backbone[2]
        if mc_index == len(backbone) - 1:
            return backbone[-2], backbone[-3]
        return backbone[mc_index - 1], backbone[mc_index + 1]

    center = backbone[mc_index]
    candidates = [index for index in backbone if index != center]
    for index in (*candidates, center):
        candidates.extend(int(i) for i in rt.icoors_ancestors[index])
    # The first two construction points use self references; the first child
    # placed off their axis supplies the missing plane (e.g. a one-atom cap).
    candidates.extend(
        i for i, ancestors in enumerate(rt.icoors_ancestors) if ancestors[0] in backbone
    )
    references = []
    origin = rt.ideal_coords[center]
    for index in dict.fromkeys(candidates):
        vector = rt.ideal_coords[index] - origin
        if index == center or numpy.linalg.norm(vector) < 1e-8:
            continue
        if references:
            first = rt.ideal_coords[references[0]] - origin
            if numpy.linalg.norm(numpy.cross(first, vector)) > 1e-8:
                return references[0], index
        else:
            references.append(index)
    raise ValueError(
        f"{rt.name}: no noncollinear construction frame at {rt.icoors[center].name}"
    )


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
