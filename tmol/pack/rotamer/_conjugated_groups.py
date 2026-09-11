"""Which blocks pack together because a covalent bond ties them.

An amino acid with something bonded to its sidechain -- a glycan, a ligand --
cannot be packed a residue at a time: moving the sidechain moves what hangs off
it, and sampling the two independently breaks the bond that joins them. The
group is the anchor residue plus everything reachable from it through
conjugation connections, and it is packed as one unit.

The anchor supplies its own chi from its rotamer library; the attached blocks
supply theirs from their chi_samples. A free ligand, or a sugar that is not
bonded to a sidechain at all, forms no group and is left alone.
"""

import attr
import numpy
import torch

from tmol.pose._conjugated_groups import (  # noqa: F401
    ConjugatedGroup,
    blocks_in_conjugated_groups,
    find_conjugated_groups,
    lockstep_group_for_block,
)


def group_atom_context(group, pose_stack):
    """Local residue types, group atom offsets and connection partners."""
    pbt = pose_stack.packed_block_types
    types = [
        pbt.active_block_types[int(pose_stack.block_type_ind[group.pose, b])]
        for b in group.blocks
    ]
    offsets = numpy.cumsum([0] + [bt.n_atoms for bt in types])
    partners = {}
    for a, ac, b, bc in group.links:
        partners[a, ac] = (b, bc)
        partners[b, bc] = (a, ac)
    return types, offsets, partners


def resolve_group_atom(uaid, owner, block_types, offsets, partners):
    """Resolve a torsion atom into the group's concatenated atom numbering."""
    atom, conn, sep = (int(v) for v in uaid)
    if conn != -1:
        partner = partners.get((owner, conn))
        if partner is None:
            return -1
        owner, conn = partner
        downstream = block_types[owner].atom_downstream_of_conn
        if not 0 <= sep < downstream.shape[1]:
            return -1
        atom = int(downstream[conn, sep])
    return int(offsets[owner]) + atom if atom >= 0 else -1


def group_sampled_chi(
    group, pose_stack, expanded_limit, limit, library_size=1, reserve_current=False
):
    """Which chi of a group's attached blocks survive the budget.

    The whole group enumerates one product, so the budget is applied across it
    rather than to each block: seven sugars of six chi each would otherwise
    multiply out. Chi freeze from the tip of the group inward, using depth in
    the kinforest that builds the group, so the linkage torsions nearest the
    anchor -- the ones that move the most atoms -- are the last to go. The
    anchor's own chi are not here; they come from its rotamer library.

    The budget counts rotamers, not conformers. A group conformer places every
    one of its blocks, so it costs one rotamer per block, and what scoring and
    the interaction graph pay for is that total. Dividing the limit by the
    number of blocks is what keeps a seven-sugar tree the same size as a
    one-sugar one.

    Returns a list of (index within the group, ChiSamples) for the survivors.
    """
    from tmol.kinematics import block_group_kinforest_data
    from tmol.pack.rotamer._chi_budget import _budgeted_chi_samples, chi_depths

    block_types, offsets, partners = group_atom_context(group, pose_stack)
    rkd, offsets = block_group_kinforest_data(block_types, group.links, anchor=0)

    entries, depths_in, owners = [], [], []
    for i, bt in enumerate(block_types):
        if i == 0:
            continue  # the anchor samples from its library, not from chi_samples
        for cs in bt.chi_samples:
            if cs.is_proton:
                continue
            entries.append(cs)
            owners.append(i)
            depths_in.append(
                resolve_group_atom(
                    bt.torsion_to_uaids[cs.chi_dihedral][2],
                    i,
                    block_types,
                    offsets,
                    partners,
                )
            )

    depths = chi_depths(rkd, depths_in)
    # every block of the group carries a copy of each conformer
    n_blocks = max(len(group.blocks), 1)
    kept = _budgeted_chi_samples(
        entries,
        depths,
        max(expanded_limit // n_blocks - int(reserve_current), 0),
        max(limit // n_blocks - int(reserve_current), 0),
        library_size=library_size,
    )

    return [(owners[index], cs) for index, cs in kept]


def protect_conjugated_anchors(task, pose_stack, exclude=()):
    """Keep samplers from moving an anchor's sidechain on its own.

    Used where the point is not to repack -- filling in missing density, say --
    but the covalent bond still has to survive. Moving the anchor without
    moving what is bonded to it would pull the bond apart.
    """
    from tmol.pack.rotamer._fallback_sampler import FallbackSampler
    from tmol.pack.rotamer._include_current_sampler import IncludeCurrentSampler

    keep = (FallbackSampler, IncludeCurrentSampler)
    groups = find_conjugated_groups(pose_stack)
    if not groups:
        return

    anchors = numpy.zeros((pose_stack.n_poses, pose_stack.max_n_blocks), dtype=bool)
    for group in groups:
        anchors[group.pose, group.anchor] = True
    mask = torch.tensor(anchors, dtype=torch.bool, device=pose_stack.device)
    for other in list(task.conformer_samplers):
        if other in exclude or isinstance(other, keep):
            continue
        task.disable_sampler_by_block_mask(other, mask)


def add_conjugated_group_sampler(task, pose_stack, sampler=None, exclude=()):
    """Attach the group sampler and keep other samplers off the anchors.

    A group is sampled as one unit. When a rotamer library is available the
    anchor is sampled along with it, its library chi multiplying the tree's
    conformers; otherwise the anchor keeps the conformation it came in with.
    Either way no other sampler may move it on its own, since that would carry
    the bonded residue along in the structure but not in the rotamer.
    """
    from tmol.pack.rotamer._conjugated_chi_sampler import ConjugatedChiSampler
    from tmol.pack.rotamer._fallback_sampler import FallbackSampler
    from tmol.pack.rotamer._include_current_sampler import IncludeCurrentSampler

    groups = find_conjugated_groups(pose_stack)
    if not groups:
        return None

    if sampler is None:
        library = next(
            (
                s
                for s in task.conformer_samplers
                if type(s).__name__ == "DunbrackChiSampler"
            ),
            None,
        )
        sampler = ConjugatedChiSampler(library_sampler=library)
    with_anchor = sampler.library_sampler is not None

    sampled = numpy.zeros((pose_stack.n_poses, pose_stack.max_n_blocks), dtype=bool)
    anchors = numpy.zeros_like(sampled)
    for group in groups:
        anchors[group.pose, group.anchor] = True
        for block in group.blocks if with_anchor else group.blocks[1:]:
            sampled[group.pose, block] = True

    def _mask(arr):
        return torch.tensor(arr, dtype=torch.bool, device=pose_stack.device)

    task.add_conformer_sampler_by_block_mask(sampler, _mask(sampled))

    # a sampler that hands back the pose's own conformation is only harmless on
    #    an anchor the group does not sample: once the group samples it, that
    #    extra rotamer would put the anchor out of step with its members
    keep = () if with_anchor else (FallbackSampler, IncludeCurrentSampler)
    for other in list(task.conformer_samplers):
        if other is sampler or other in exclude or isinstance(other, keep):
            continue
        task.disable_sampler_by_block_mask(other, _mask(anchors))
    return sampler


@attr.s(auto_attribs=True, frozen=True, slots=True)
class GroupCollapse:
    """How a group's members were folded into one choice for the packer.

    The packer is handed a smaller rotamer set than was built: a member's
    rotamers past the first are gone, and every reference to them has been sent
    to the representative's corresponding rotamer, so the packer chooses once
    for the whole group and cannot pick conformer 5 of one sugar against
    conformer 200 of the next. The tensors named ``compact_`` describe that
    smaller set and are what the interaction graph is built from;
    ``rot_offset_for_block`` and ``members`` stay in the original indexing,
    which is what the rotamer set's coordinates are addressed by.
    """

    # original rotamer index -> its index in the compacted set, with a member's
    #    rotamers pointing at the representative's
    rot_map: torch.Tensor
    # compacted rotamer index -> the original rotamer it stands for
    compact_to_orig: torch.Tensor
    compact_n_rots_for_block: torch.Tensor
    compact_rot_offset_for_block: torch.Tensor
    compact_block_ind_for_rot: torch.Tensor
    compact_pose_for_rot: torch.Tensor
    compact_block_type_ind_for_rot: torch.Tensor
    compact_n_rots_for_pose: torch.Tensor
    compact_rot_offset_for_pose: torch.Tensor
    # original indexing, for writing the chosen conformer back out
    rot_offset_for_block: torch.Tensor
    n_rots_for_block: torch.Tensor
    # (pose, representative block, [(member block, that member's rot offset)])
    members: tuple


def collapse_group_rotamers(pose_stack, rotamer_set, groups):
    """Fold each group's members into a single block for the packer.

    Returns the replacement bookkeeping, or None when there is nothing to fold.
    """
    if not groups:
        return None

    device = rotamer_set.coords.device
    n_rots = rotamer_set.block_ind_for_rot.shape[0]
    rot_map = torch.arange(n_rots, dtype=torch.int64, device=device)
    keep = torch.ones((n_rots,), dtype=torch.bool, device=device)
    n_rots_for_block = rotamer_set.n_rots_for_block.clone()
    rot_offset = rotamer_set.rot_offset_for_block
    group_of_rot = torch.full((n_rots,), -1, dtype=torch.int64, device=device)
    conformer_of_rot = torch.full((n_rots,), -1, dtype=torch.int64, device=device)

    members = []
    for gi, group in enumerate(groups):
        counts = [int(n_rots_for_block[group.pose, b]) for b in group.blocks]
        sampled = [b for b, c in zip(group.blocks, counts) if c > 1]
        if len(sampled) < 2:
            continue  # nothing to correlate
        n_conf = int(n_rots_for_block[group.pose, sampled[0]])
        if any(int(n_rots_for_block[group.pose, b]) != n_conf for b in sampled):
            raise ValueError(
                f"group at anchor {group.anchor} has members with differing "
                "rotamer counts; they cannot be in lockstep"
            )

        rep = sampled[0]
        rep_off = int(rot_offset[group.pose, rep])
        entry = []
        for b in sampled:
            off = int(rot_offset[group.pose, b])
            entry.append((b, off))
            idx = torch.arange(off, off + n_conf, dtype=torch.int64, device=device)
            group_of_rot[idx] = gi
            conformer_of_rot[idx] = torch.arange(
                n_conf, dtype=torch.int64, device=device
            )
            if b == rep:
                continue
            rot_map[idx] = torch.arange(
                rep_off, rep_off + n_conf, dtype=torch.int64, device=device
            )
            # One rotamer, not none: a real block with no rotamers is neither
            #    molten nor background, and the packer's bookkeeping needs it to
            #    be one or the other. The one it keeps carries no energy -- every
            #    reference to this member was remapped onto the representative --
            #    and its coordinates are overwritten once the group's conformer
            #    is chosen, so it is a placeholder and nothing more.
            keep[idx[1:]] = False
            n_rots_for_block[group.pose, b] = 1
        members.append((group.pose, rep, tuple(entry)))

    if not members:
        return None

    # Counts and offsets have to keep describing a partition of the rotamer
    #    range: the interaction graph recovers a block's rotamer count by
    #    differencing consecutive blocks' offsets, so a rotamer left sitting in
    #    a gap is silently charged to whichever block precedes it.
    # Include the exclusive end: trailing zero-rotamer blocks and empty poses
    # can have an offset equal to n_rots.
    orig_to_compact = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=device), keep.cumsum(0))
    )
    compact_to_orig = torch.nonzero(keep).view(-1)

    compact_rot_offset_for_block = torch.where(
        rot_offset >= 0,
        orig_to_compact[rot_offset.clamp_min(0).to(torch.int64)],
        rot_offset,
    )
    compact_rot_offset_for_pose = orig_to_compact[
        rotamer_set.rot_offset_for_pose.to(torch.int64)
    ]
    n_rots_for_pose = torch.zeros_like(rotamer_set.n_rots_for_pose)
    n_poses = n_rots_for_pose.shape[0]
    for pose in range(n_poses):
        start = int(rotamer_set.rot_offset_for_pose[pose])
        stop = start + int(rotamer_set.n_rots_for_pose[pose])
        n_rots_for_pose[pose] = int(keep[start:stop].sum())

    return (
        GroupCollapse(
            rot_map=orig_to_compact[rot_map],
            compact_to_orig=compact_to_orig,
            compact_n_rots_for_block=n_rots_for_block,
            compact_rot_offset_for_block=compact_rot_offset_for_block.to(
                rot_offset.dtype
            ),
            compact_block_ind_for_rot=rotamer_set.block_ind_for_rot[compact_to_orig],
            compact_pose_for_rot=rotamer_set.pose_for_rot[compact_to_orig],
            compact_block_type_ind_for_rot=rotamer_set.block_type_ind_for_rot[
                compact_to_orig
            ],
            compact_n_rots_for_pose=n_rots_for_pose,
            compact_rot_offset_for_pose=compact_rot_offset_for_pose.to(
                rotamer_set.rot_offset_for_pose.dtype
            ),
            rot_offset_for_block=rot_offset,
            n_rots_for_block=n_rots_for_block,
            members=tuple(members),
        ),
        group_of_rot,
        conformer_of_rot,
    )


def collapse_group_energies(energies, collapse, group_of_rot, conformer_of_rot):
    """Remap a group's rotamer pair energies onto its representative.

    A pair that puts two members in DIFFERENT conformers describes a structure
    the group cannot adopt, so it is dropped rather than folded in; every other
    pair is summed onto the representative by coalescing. The summing matters:
    the interaction graph assigns a one-body energy rather than accumulating
    it, so the total has to be in one entry by the time it gets there.

    The returned tensor is indexed by the compacted rotamer numbering.
    """
    energies = energies.coalesce()
    idx = energies.indices()
    vals = energies.values()

    r1, r2 = idx[1].to(torch.int64), idx[2].to(torch.int64)
    g1, g2 = group_of_rot[r1], group_of_rot[r2]
    inconsistent = (
        (g1 >= 0) & (g1 == g2) & (conformer_of_rot[r1] != conformer_of_rot[r2])
    )
    keep = torch.logical_not(inconsistent)

    new_r1 = collapse.rot_map[r1[keep]]
    new_r2 = collapse.rot_map[r2[keep]]

    # Folding members onto a representative moves rotamers to a lower block, so
    #    a pair that was in order may no longer be. The interaction graph
    #    requires block1 < block2 and DROPS anything else without complaint, so
    #    put each pair back in order; the energy is symmetric either way.
    b1 = collapse.compact_block_ind_for_rot[new_r1]
    b2 = collapse.compact_block_ind_for_rot[new_r2]
    swap = b1 > b2
    ordered_r1 = torch.where(swap, new_r2, new_r1)
    ordered_r2 = torch.where(swap, new_r1, new_r2)

    n_compact = collapse.compact_to_orig.shape[0]
    new_idx = torch.stack([idx[0][keep], ordered_r1, ordered_r2])
    return torch.sparse_coo_tensor(
        new_idx, vals[keep], size=(energies.shape[0], n_compact, n_compact)
    ).coalesce()


def write_group_members(new_pose_stack, rotamer_set, collapse, assignment):
    """Give every member of a group the conformer chosen for its representative.

    The packer decided once per group, on the representative's block; the other
    members were folded onto it and never assigned. ``assignment`` gives the
    rotamer each block ended up with, so the conformer index is read from the
    representative rather than guessed at.
    """
    coords = new_pose_stack.coords.clone()
    pbt = new_pose_stack.packed_block_types

    for pose, rep, entry in collapse.members:
        rep_off = int(collapse.rot_offset_for_block[pose, rep])
        chosen = int(assignment[pose, rep]) - rep_off
        if chosen < 0 or chosen >= int(collapse.n_rots_for_block[pose, rep]):
            raise ValueError(
                f"group representative block {rep} of pose {pose} was assigned "
                f"rotamer {int(assignment[pose, rep])}, which is not one of the "
                f"{int(collapse.n_rots_for_block[pose, rep])} it offered"
            )
        for block, off in entry:
            if block == rep:
                continue
            n = int(pbt.n_atoms[int(new_pose_stack.block_type_ind[pose, block])])
            start = int(new_pose_stack.block_coord_offset[pose, block])
            src = int(rotamer_set.coord_offset_for_rot[off + chosen])
            coords[pose, start : start + n] = rotamer_set.coords[src : src + n]

    return attr.evolve(new_pose_stack, coords=coords)
