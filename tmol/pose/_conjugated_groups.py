"""Which blocks pack together because a covalent bond ties them.

An amino acid with something bonded to its sidechain -- a glycan, a ligand --
cannot be packed a residue at a time: moving the sidechain moves what hangs off
it, and sampling the two independently breaks the bond that joins them. The
group is the anchor residue plus everything reachable from it through
conjugation connections, and it is packed as one unit.

This lives beside PoseStack rather than in the packer because scoring needs it
too: a group's members are sampled in lockstep, so the rotamer-pair enumerator
has to know which blocks move together in order not to enumerate combinations
the group cannot adopt.
"""

import attr
import numpy
import torch

from typing import List, Tuple

from tmol.pose._pose_stack import PoseStack
from tmol.pose._packed_block_types import (
    annotate_packed_block_types_w_conjugation_conns,
)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class ConjugatedGroup:
    """One anchor residue and the blocks bonded to its sidechain.

    ``blocks`` lists the anchor first, then the attached blocks in the order a
    breadth-first walk out from the anchor reaches them, so a block always
    follows the one it is bonded to. ``links`` gives each bond as (parent index
    within ``blocks``, connection on the parent, child index within ``blocks``,
    connection on the child) -- the form block_group_kinforest_data takes.
    """

    pose: int
    blocks: Tuple[int, ...]
    links: Tuple[Tuple[int, int, int, int], ...]

    @property
    def anchor(self) -> int:
        return self.blocks[0]

    def __len__(self) -> int:
        return len(self.blocks)


def find_conjugated_groups(pose_stack: PoseStack) -> List[ConjugatedGroup]:
    """Every anchor-plus-attachments group in the stack."""
    pbt = pose_stack.packed_block_types
    annotate_packed_block_types_w_conjugation_conns(pbt)

    # most structures have no conjugations at all; answer them without walking
    #    the pose, since this runs on every pack
    if not bool(pbt.conjugation_conn.any()):
        return []

    conj = pbt.conjugation_conn.cpu().numpy()
    n_conn = pbt.n_conn.cpu().numpy()
    up = pbt.up_conn_inds.cpu().numpy()
    down = pbt.down_conn_inds.cpu().numpy()
    bti = pose_stack.block_type_ind.cpu().numpy()
    irc = pose_stack.inter_residue_connections.cpu().numpy()

    # an anchor is a polymer residue carrying a conjugation; find the candidates
    #    with a gather rather than a scan over every block
    is_polymer = (up >= 0) | (down >= 0)
    has_conj = conj.any(axis=1)
    candidate_bt = is_polymer & has_conj
    real = bti >= 0
    candidates = numpy.logical_and(real, candidate_bt[numpy.where(real, bti, 0)])

    groups = []
    for pose in range(pose_stack.n_poses):
        claimed = set()
        for block in numpy.nonzero(candidates[pose])[0]:
            block = int(block)
            if block in claimed:
                continue

            blocks = [block]
            links = []
            index_of = {block: 0}
            queue = [block]
            while queue:
                cur = queue.pop(0)
                cur_bt = int(bti[pose, cur])
                for c in range(int(n_conn[cur_bt])):
                    if not conj[cur_bt, c]:
                        continue
                    partner = int(irc[pose, cur, c, 0])
                    if partner < 0 or int(bti[pose, partner]) < 0:
                        continue
                    if partner in index_of:
                        continue  # already in the group; a cycle closes here
                    index_of[partner] = len(blocks)
                    blocks.append(partner)
                    links.append(
                        (
                            index_of[cur],
                            c,
                            index_of[partner],
                            int(irc[pose, cur, c, 1]),
                        )
                    )
                    queue.append(partner)

            if len(blocks) > 1:
                claimed.update(blocks)
                groups.append(
                    ConjugatedGroup(pose=pose, blocks=tuple(blocks), links=tuple(links))
                )
    return groups


def blocks_in_conjugated_groups(pose_stack: PoseStack) -> torch.Tensor:
    """Per-block mask of everything a conjugated group covers."""
    mask = numpy.zeros((pose_stack.n_poses, pose_stack.max_n_blocks), dtype=bool)
    for group in find_conjugated_groups(pose_stack):
        for block in group.blocks:
            mask[group.pose, block] = True
    return torch.tensor(mask, dtype=torch.bool, device=pose_stack.device)


def lockstep_group_for_block(pose_stack, rotamer_set) -> torch.Tensor:
    """Per-block id shared by the blocks a group samples in lockstep.

    Rotamer k of every block carrying the same id is group conformer k, so a
    pair of these blocks only ever coexists at matching rotamer indices. Blocks
    outside such a group get -1. Only blocks the packer actually samples are
    marked, and only where the group has more than one of them; a group with a
    single sampled block has nothing to stay in step with.
    """
    n_rots = rotamer_set.n_rots_for_block
    out = torch.full(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        -1,
        dtype=torch.int32,
        device=n_rots.device,
    )
    for gi, group in enumerate(find_conjugated_groups(pose_stack)):
        sampled = [b for b in group.blocks if int(n_rots[group.pose, b]) > 1]
        if len(sampled) < 2:
            continue
        for b in sampled:
            out[group.pose, b] = gi
    return out
