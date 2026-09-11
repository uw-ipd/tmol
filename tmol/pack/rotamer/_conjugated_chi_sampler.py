"""Rotamers for the blocks bonded to an amino acid's sidechain.

A glycan or a ligand joined to a sidechain cannot be sampled a residue at a
time: attachment torsions can have central atoms in different residues, so a
single-block kinforest cannot turn all of their bonds. The group --
anchor plus everything bonded to it -- is enumerated as one unit instead, with
the anchor's own chi coming from its rotamer library and the attached blocks'
chi from their chi_samples.

The group owns its members' coupled choices, including the anchor's library
states. Independent torsions preserve cyclic cores and external attachments;
this does not supply alternate ring puckers or a ring-closure solver.
"""

import attr
from collections import OrderedDict
from copy import copy
import itertools
import math
import numpy
import torch

from typing import Tuple

from tmol.types import Tensor, validate_args
from tmol.chemical import RefinedResidueType
from tmol.pose import PackedBlockTypes, PoseStack
from tmol.pose._util import get_named_torsions
from tmol.pack.rotamer._chi_sampler import ChiSampler
from tmol.pack.rotamer._conformer_sampler import sc_roots_for_chis
from tmol.pack.rotamer._conjugated_groups import (
    find_conjugated_groups,
    group_sampled_chi,
    group_sampling_topology,
    resolve_group_atom,
    GroupSamplingPlan,
)
from tmol.pack._packer_task import (
    DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT,
    DEFAULT_CHI_SAMPLE_LIMIT,
)


def _heavy_chi(rt: RefinedResidueType):
    """The chi this sampler is responsible for: sampled, and not a proton chi."""
    return [cs for cs in rt.chi_samples if not cs.is_proton]


def _has_conjugation(rt):
    return any(
        connection.name not in ("up", "down", "dslf") for connection in rt.connections
    )


# Bound each native transform workspace; a single larger group is indivisible.
_KINEMATICS_BATCH_ATOMS = 4096


def _repeated_kinforest(rot_kf, count, device):
    """Independent copies of a local tree, sharing only the global root."""
    n_atoms = len(rot_kf.id)
    local = numpy.stack(
        [
            rot_kf.id,
            rot_kf.doftype,
            rot_kf.parent + 1,
            rot_kf.frame_x + 1,
            rot_kf.frame_y + 1,
            rot_kf.frame_z + 1,
        ],
        axis=1,
    )
    repeated = numpy.broadcast_to(local, (count, n_atoms, 6)).copy()
    offsets = numpy.arange(count, dtype=numpy.int32)[:, None] * n_atoms
    repeated[:, :, 0] += offsets
    repeated[:, :, 2:] += numpy.where(local[None, :, 2:] > 0, offsets[:, :, None], 0)
    root = numpy.array([[-1, 0, 0, 0, 0, 0]], dtype=numpy.int32)
    return torch.tensor(
        numpy.concatenate([root, repeated.reshape(-1, 6)]),
        dtype=torch.int32,
        device=device,
    )


def _fold_group_conformers(rot_kf, dofs):
    """Fold conformers in bounded scans; return original atom order."""
    from tmol.kinematics.compiled import forward_only_op

    count, n_nodes = dofs.shape[:2]
    n_atoms = n_nodes - 1
    if count == 0:
        return dofs.new_empty((0, n_atoms, 3))
    chunk = max(1, _KINEMATICS_BATCH_ATOMS // n_atoms)
    if count > chunk:
        return torch.cat(
            [
                _fold_group_conformers(rot_kf, dofs[start : start + chunk])
                for start in range(0, count, chunk)
            ]
        )
    offsets = numpy.arange(count, dtype=numpy.int32)[:, None] * n_atoms
    node_parts, scan_parts = [], []
    for (n0, s0), (n1, s1) in zip(rot_kf.gens[:-1], rot_kf.gens[1:]):
        nodes = rot_kf.nodes[n0:n1]
        node_parts.append((nodes + numpy.where(nodes > 0, offsets, 0)).ravel())
        scan_parts.append(
            (rot_kf.scans[s0:s1] + numpy.arange(count)[:, None] * (n1 - n0)).ravel()
        )
    stack = _repeated_kinforest(rot_kf, count, dofs.device)
    coords = forward_only_op(
        torch.cat([dofs[0, :1], dofs[:, 1:].reshape(-1, 9)]),
        torch.tensor(
            numpy.concatenate(node_parts), dtype=torch.int32, device=dofs.device
        ),
        torch.tensor(
            numpy.concatenate(scan_parts), dtype=torch.int32, device=dofs.device
        ),
        torch.tensor(rot_kf.gens * count, dtype=torch.int32, device="cpu"),
        stack,
    )
    original_order = torch.tensor(numpy.argsort(rot_kf.id), device=dofs.device)
    return coords[1:].reshape(count, n_atoms, 3)[:, original_order]


def _measure_member_dofs(rot_kf, coords):
    """Measure copies of one member in bounded inverse-kinematics batches."""
    from tmol.kinematics.compiled import inverse_kin

    count, n_atoms = coords.shape[:2]
    if count == 0:
        return coords.new_empty((0, n_atoms, 9))
    chunk = max(1, _KINEMATICS_BATCH_ATOMS // n_atoms)
    if count > chunk:
        return torch.cat(
            [
                _measure_member_dofs(rot_kf, coords[start : start + chunk])
                for start in range(0, count, chunk)
            ]
        )
    stack = _repeated_kinforest(rot_kf, count, coords.device)
    atom_order = torch.tensor(rot_kf.id, dtype=torch.int64, device=coords.device)
    ordered = torch.cat(
        [coords.new_zeros((1, 3)), coords[:, atom_order].reshape(-1, 3)]
    )
    dofs = inverse_kin(
        ordered, stack[:, 2], stack[:, 3], stack[:, 4], stack[:, 5], stack[:, 1]
    )
    return dofs[1:].reshape(count, n_atoms, 9)


@attr.s(auto_attribs=True)
class ConjugatedChiSampler(ChiSampler):
    chi_sample_expanded_limit: int = DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT
    chi_sample_limit: int = DEFAULT_CHI_SAMPLE_LIMIT
    # the sampler that supplies an anchor's own chi; without it the anchor
    #    is held at the conformation the pose came in with
    library_sampler: object = None
    # offer the conformation the pose came in with, as a single block does
    include_current: bool = True

    @classmethod
    def sampler_name(cls):
        return "ConjugatedChiSampler"

    @validate_args
    def defines_rotamers_for_rt(self, rt: RefinedResidueType):
        # a type-level necessary condition; which blocks are actually sampled is
        #    decided per pose, since being bonded to a sidechain is not a
        #    property of the residue type
        return _has_conjugation(rt)

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        builds = torch.tensor(
            [_has_conjugation(bt) for bt in pbt.active_block_types],
            dtype=torch.bool,
            device=pbt.device,
        )
        return builds[bt_inds]

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return sc_roots_for_chis(rt, [cs.chi_dihedral for cs in _heavy_chi(rt)])

    def group_conformers(
        self, pose_stack: PoseStack, anchor_chi=None, budget=None, *, plans=None
    ):
        """Enumerate each group's product using actual library cardinality.

        Limits count all member rotamers, including the offered input state.
        Distinct library states projected onto movable axes are retained; an
        impossible limit fails before allocating the Cartesian product.
        Constrained/frozen chi retain their input geometry.
        """
        expanded_limit, limit = budget or (
            self.chi_sample_expanded_limit,
            self.chi_sample_limit,
        )
        out = []
        pbt = pose_stack.packed_block_types
        if plans is None:
            first_owner = int(self.library_sampler is None)
            plans = [
                GroupSamplingPlan(
                    group,
                    group_sampling_topology(
                        group, pose_stack, (0,) if first_owner else ()
                    ),
                    tuple(range(first_owner, len(group))),
                )
                for group in find_conjugated_groups(pose_stack)
                if len(group) > first_owner
            ]
        for sampling_plan in plans:
            group, topology = sampling_plan.group, sampling_plan.topology
            block_types, offsets, partners = (
                topology.block_types,
                topology.offsets,
                topology.partners,
            )
            anchor_cols = []
            lib = numpy.empty((1, 0), dtype=numpy.float32)
            entry = (anchor_chi or {}).get((group.pose, group.anchor))
            if entry is not None and entry[0].shape[0]:
                anchor_atoms, anchor_values = entry
                anchor_bt = pbt.active_block_types[
                    int(pose_stack.block_type_ind[group.pose, group.anchor])
                ]
                atom_to_chi = {
                    int(uaids[2][0]): (name, uaids[1][0], uaids[2][0])
                    for name, uaids in anchor_bt.torsion_to_uaids.items()
                    if name.startswith("chi") and uaids[1][0] >= 0 and uaids[2][0] >= 0
                }
                keep_cols = []
                for j in range(anchor_atoms.shape[1]):
                    atom = int(anchor_atoms[0, j])
                    if atom < 0 or atom not in atom_to_chi:
                        continue
                    name, ab, ac = atom_to_chi[atom]
                    axis = frozenset((int(ab), int(ac)))
                    if axis not in topology.movable_axes or any(
                        axis == frozenset(c[2:]) for c in anchor_cols
                    ):
                        continue
                    anchor_cols.append((0, name, ab, ac))
                    keep_cols.append(j)
                if anchor_cols:
                    lib = anchor_values[:, keep_cols]
                    if len(keep_cols) < anchor_values.shape[1]:
                        # Projecting away constrained angles can make library
                        # rows identical. Preserve the first occurrence order.
                        _, first = numpy.unique(lib, axis=0, return_index=True)
                        lib = lib[numpy.sort(first)]

            kept = group_sampled_chi(
                group,
                pose_stack,
                expanded_limit,
                limit,
                library_size=lib.shape[0],
                reserve_current=self.include_current,
                topology=topology,
                exclude_axes={frozenset(c[2:]) for c in anchor_cols},
                include_anchor=self.library_sampler is not None,
                sampled_members=len(sampling_plan.owners),
            )
            columns, per_chi = [], []
            for owner, cs in kept:
                bt = pbt.active_block_types[
                    int(pose_stack.block_type_ind[group.pose, group.blocks[owner]])
                ]
                uaids = bt.torsion_to_uaids[cs.chi_dihedral]
                axis = [
                    resolve_group_atom(u, owner, block_types, offsets, partners)
                    for u in uaids[1:3]
                ]
                if min(axis) < 0:
                    raise ValueError(f"Cannot resolve group torsion {cs.chi_dihedral}")
                # Central atoms use group-wide numbering, including connections.
                columns.append((owner, cs.chi_dihedral, *axis))
                values = []
                for value in cs.samples:
                    values.append(value)
                    for expansion in cs.expansions:
                        values.extend((value - expansion, value + expansion))
                per_chi.append([math.radians(v) for v in values])
            # The empty product is one state, not an absent group: the anchor
            # can still move while all child chi are frozen.
            tree = numpy.array(list(itertools.product(*per_chi)), dtype=numpy.float32)
            conformers = numpy.concatenate(
                [numpy.repeat(lib, len(tree), axis=0), numpy.tile(tree, (len(lib), 1))],
                axis=1,
            )
            columns = anchor_cols + columns
            out.append(
                (
                    group,
                    columns,
                    self._with_current(pose_stack, group, columns, conformers),
                )
            )
        return out

    def _with_current(self, pose_stack: PoseStack, group, columns, conformers):
        """Offer the conformation the group came in with, as one more conformer.

        A single block keeps its current rotamer when the packer is restricted
        to repacking; a group has to do the same or the structure it started
        from is not among the choices, and packing can only move it away.
        """
        if not self.include_current or not columns:
            return conformers
        current = numpy.empty((1, len(columns)), dtype=numpy.float32)
        for j, (owner, name, _b, _c) in enumerate(columns):
            value = get_named_torsions(
                pose_stack,
                poses=group.pose,
                blocks=group.blocks[owner],
                names=name,
                degrees=False,
            )
            if not math.isfinite(value):
                return conformers  # cannot measure it; leave the grid alone
            current[0, j] = value
        return numpy.concatenate([conformers, current], axis=0)

    def _group_kinforest(self, pose_stack: PoseStack, group):
        """The kinforest that folds one group, cached on its shape."""
        from tmol.pack.rotamer._single_residue_kinforest import (
            construct_block_group_kinforest,
        )

        pbt = pose_stack.packed_block_types
        types = tuple(
            pose_stack.block_type_ind[group.pose, list(group.blocks)].tolist()
        )
        key = (types, group.links)
        cache = getattr(pbt, "conjugated_kinforest_cache", None)
        if cache is None:
            cache = OrderedDict()
            setattr(pbt, "conjugated_kinforest_cache", cache)
        if key not in cache:
            topology = group_sampling_topology(group, pose_stack)
            cache[key] = construct_block_group_kinforest(
                topology.block_types,
                group.links,
                anchor=0,
                kinforest_data=(topology.kinforest, topology.offsets),
            )
            if len(cache) > 32:
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def group_coords(self, pose_stack: PoseStack, group, columns, conformers):
        """Coordinates of every atom of the group, one set per conformer.

        The group folds as one unit, so a linkage torsion carries the blocks
        beyond it: that is the whole point of building here rather than a block
        at a time. Returns conformers x group-atoms x 3.
        """
        from tmol.kinematics.compiled import inverse_kin
        from tmol.kinematics import KinForest

        rot_kf, offsets = self._group_kinforest(pose_stack, group)
        device = pose_stack.device

        def _t(x):
            return torch.tensor(numpy.asarray(x, dtype=numpy.int32), device=device)

        kinforest = KinForest(
            id=_t(numpy.concatenate([[-1], rot_kf.id])),
            doftype=_t(numpy.concatenate([[0], rot_kf.doftype])),
            parent=_t(numpy.concatenate([[0], rot_kf.parent + 1])),
            frame_x=_t(numpy.concatenate([[0], rot_kf.frame_x + 1])),
            frame_y=_t(numpy.concatenate([[0], rot_kf.frame_y + 1])),
            frame_z=_t(numpy.concatenate([[0], rot_kf.frame_z + 1])),
        )

        # the group's atoms, in the order the group kinforest numbers them
        pbt = pose_stack.packed_block_types
        flat = []
        for i, b in enumerate(group.blocks):
            n = int(pbt.n_atoms[int(pose_stack.block_type_ind[group.pose, b])])
            start = int(pose_stack.block_coord_offset[group.pose, b])
            flat.append(pose_stack.coords[group.pose, start : start + n])
        group_coords = torch.cat(flat, dim=0)

        kf_coords = torch.cat(
            [torch.zeros((1, 3), dtype=torch.float32, device=device), group_coords]
        )[kinforest.id.to(torch.int64) + 1]
        kf_coords[0, :] = 0
        dofs = inverse_kin(
            kf_coords,
            kinforest.parent,
            kinforest.frame_x,
            kinforest.frame_y,
            kinforest.frame_z,
            kinforest.doftype,
        )

        n_conf = conformers.shape[0]
        stacked = dofs.unsqueeze(0).repeat(n_conf, 1, 1)
        kfo_for_atom = numpy.full(int(offsets[-1]) + 1, -1, dtype=numpy.int64)
        for kfo, orig in enumerate(rot_kf.id):
            kfo_for_atom[int(orig)] = kfo + 1

        # A kinforest's phi_c is not the dihedral itself -- its zero lies
        #    elsewhere -- so a measured angle cannot be written into it
        #    directly. Applying the CHANGE in angle sidesteps the offset.
        from tmol.pose._util import _measure_torsions, _torsion_requests

        parent_of = {int(k) + 1: int(p) + 1 for k, p in enumerate(rot_kf.parent)}
        for col, (owner, chi_name, atom_b, atom_c) in enumerate(columns):
            # phi_c turns about parent->child, so the bond's CHILD carries the
            #    torsion; which of the two that is depends on how this group's
            #    tree runs, not on the order the torsion names them
            nb = int(kfo_for_atom[int(atom_b)])
            nc = int(kfo_for_atom[int(atom_c)])
            if parent_of.get(nc) == nb:
                node = nc
            elif parent_of.get(nb) == nc:
                node = nb
            else:
                raise ValueError(
                    f"{chi_name} of group block {owner} turns a bond that this "
                    "group's kinforest does not contain"
                )
            block = group.blocks[owner]
            current = float(
                _measure_torsions(
                    pose_stack,
                    _torsion_requests(pose_stack, group.pose, block, chi_name),
                    degrees=False,
                )[0]
            )
            target = torch.tensor(
                conformers[:, col], dtype=torch.float32, device=device
            )
            stacked[:, node, 3] = dofs[node, 3] + (target - current)
        return kinforest, stacked, offsets

    def create_samples_for_poses(self, pose_stack: PoseStack, task):
        """One rotamer per attached block per group conformer.

        Every active member gets the same number of rotamers, and rotamer k
        of each member belongs to group conformer k. That correspondence is what
        keeps them in step; it is recorded so the packer can later treat a
        group's rotamers as one choice rather than several. Inactive members
        constrain the group's geometry and retain their input via fallback.
        """
        device = pose_stack.device
        n_gbt = task.cons_bt_pose.shape[0]
        n_rots_for_gbt = torch.zeros(n_gbt, dtype=torch.int32, device=device)

        pose_of = task.cons_bt_pose.cpu().numpy()
        block_of = task.cons_bt_block.cpu().numpy()
        bt_of = task.cons_bt_block_type.cpu().numpy()
        gbt_for = {}
        allowed = task.is_cons_bt_allowed.cpu().numpy()
        sampler_index = task.conformer_sampler_index[id(self)]
        sampler_allowed = (
            task.per_block_conformer_sampler_allowed[:, :, sampler_index].cpu().numpy()
        )
        block_types = pose_stack.block_type_ind.cpu().numpy()
        for i in range(n_gbt):
            if allowed[i] and sampler_allowed[pose_of[i], block_of[i]]:
                gbt_for[(int(pose_of[i]), int(block_of[i]), int(bt_of[i]))] = i

        plans = []
        for group in find_conjugated_groups(pose_stack):
            first_owner = int(self.library_sampler is None)
            owners = tuple(
                owner
                for owner in range(first_owner, len(group))
                if (
                    group.pose,
                    group.blocks[owner],
                    int(block_types[group.pose, group.blocks[owner]]),
                )
                in gbt_for
            )
            if not owners:
                continue
            fixed = tuple(owner for owner in range(len(group)) if owner not in owners)
            plans.append(
                GroupSamplingPlan(
                    group, group_sampling_topology(group, pose_stack, fixed), owners
                )
            )

        anchor_chi = self.anchor_library_chi(
            pose_stack,
            task,
            [plan.group for plan in plans],
            plans=plans,
        )
        groups = self.group_conformers(
            pose_stack,
            anchor_chi,
            budget=getattr(task, "chi_sample_budget", None),
            plans=plans,
        )
        emitted = []
        for gi, (group, columns, conformers) in enumerate(groups):
            n_conf = int(conformers.shape[0])
            # Planning has already checked both task masks for each owner.
            for owner in plans[gi].owners:
                block = group.blocks[owner]
                bt = int(block_types[group.pose, block])
                gbt = gbt_for[group.pose, block, bt]
                n_rots_for_gbt[gbt] = n_conf
                emitted.append((gbt, gi, owner, n_conf))

        # rotamers must be listed in ascending block-type order: the merge
        #    locates a sampler's rotamers by a cumulative sum over the counts
        #    above, so any other order silently points them at the wrong
        #    conformers
        emitted.sort()
        gbt_for_rotamer, plan = [], []
        correlated_gbts = [[] for _ in groups]
        for gbt, gi, owner, n_conf in emitted:
            plan.append((gi, owner, gbt, len(gbt_for_rotamer)))
            correlated_gbts[gi].append(gbt)
            gbt_for_rotamer.extend([gbt] * n_conf)

        return (
            n_rots_for_gbt,
            torch.tensor(gbt_for_rotamer, dtype=torch.int32, device=device),
            dict(
                groups=groups,
                plan=plan,
                correlated_gbts=tuple(tuple(gbts) for gbts in correlated_gbts),
            ),
        )

    def fill_dofs_for_samples(
        self,
        pose_stack,
        task,
        orig_kinforest,
        orig_dofs_kto,
        gbt_for_conformer,
        block_type_ind_for_conformer,
        n_dof_atoms_offset_for_conformer,
        conformer_built_by_sampler,
        conf_inds_for_sampler,
        sampler_n_rots_for_gbt,
        sampler_gbt_for_rotamer,
        sample_dict,
        conf_dofs_kto,
    ):
        """Write each member's dofs from the coordinates the group folds to.

        The group is folded once per conformer through its own kinforest, and
        each member's share of the result is measured back into that member's
        own kinforest. Going through coordinates is what carries the correlation
        across the bond: a member's placement comes from where the group put it,
        not from where it started.
        """
        from tmol.pack.rotamer._single_residue_kinforest import (
            construct_single_residue_kinforest,
        )

        pbt = pose_stack.packed_block_types
        # One transfer for all destinations, not two scalar synchronizations
        # for every member/conformer. Validate before indexing native outputs.
        conf_indices = conf_inds_for_sampler.cpu().numpy()
        dof_offsets = n_dof_atoms_offset_for_conformer.cpu().numpy()
        if numpy.any((conf_indices < 0) | (conf_indices >= len(dof_offsets))):
            raise IndexError("conjugated rotamer maps to an invalid global conformer")
        destinations = dof_offsets[conf_indices] + 1
        for gi, (group, columns, conformers) in enumerate(sample_dict["groups"]):
            _, dofs, offsets = self.group_coords(pose_stack, group, columns, conformers)
            rot_kf, _ = self._group_kinforest(pose_stack, group)
            folded = _fold_group_conformers(rot_kf, dofs)
            for pgi, owner, gbt, first in sample_dict["plan"]:
                if pgi != gi:
                    continue
                block = group.blocks[owner]
                bt = pbt.active_block_types[
                    int(pose_stack.block_type_ind[group.pose, block])
                ]
                construct_single_residue_kinforest(bt)
                lo, hi = int(offsets[owner]), int(offsets[owner + 1])
                member_dofs = _measure_member_dofs(
                    bt.rotamer_kinforest, folded[:, lo:hi]
                )
                starts = destinations[first : first + len(folded)]
                if len(starts) != len(folded) or numpy.any(
                    (starts < 1) | (starts + hi - lo > conf_dofs_kto.shape[0])
                ):
                    raise IndexError(
                        f"conjugated group block {block} maps outside the global dof rows"
                    )
                rows = torch.tensor(
                    starts[:, None] + numpy.arange(hi - lo),
                    dtype=torch.int64,
                    device=pose_stack.device,
                )
                conf_dofs_kto[rows] = member_dofs

    def anchor_library_chi(self, pose_stack, task, groups, *, plans=None):
        """The chi values a rotamer library offers each group's anchor.

        The anchor is an ordinary amino acid, so its chi come from its library
        rather than from chi_samples -- but they have to be enumerated HERE,
        together with the tree, or the two would be chosen independently and the
        bond would not survive. The library sampler is asked for them directly;
        it is kept from emitting rotamers of its own for these blocks.
        """
        if self.library_sampler is None:
            return {}
        index = task.conformer_sampler_index[id(self.library_sampler)]
        # Only anchors need this extra library pass. Keep the caller's task
        # immutable, including while the library is executing or raises.
        library_task = copy(task)
        # The group's budget applies after projecting out constrained angles.
        # It is not a cap on the temporary source-library rows used to obtain
        # those distinct projected states.
        library_task.chi_sample_budget = None
        allowed = torch.zeros_like(task.per_block_conformer_sampler_allowed)
        library_task.per_block_conformer_sampler_allowed = allowed
        for group_index, group in enumerate(groups):
            topology = (
                plans[group_index].topology
                if plans is not None
                else group_sampling_topology(group, pose_stack)
            )
            if any(
                name.startswith("chi")
                and frozenset((int(u[1][0]), int(u[2][0]))) in topology.movable_axes
                for name, u in topology.block_types[0].torsion_to_uaids.items()
            ):
                allowed[group.pose, group.anchor, index] = True
        if not bool(allowed.any()):
            return {}
        _n, gbt_for_rot, chi_atoms, chi = self.library_sampler.sample_chi_for_poses(
            pose_stack, library_task
        )

        pose_of = task.cons_bt_pose.cpu().numpy()
        block_of = task.cons_bt_block.cpu().numpy()
        bt_of = task.cons_bt_block_type.cpu().numpy()
        gbt_np = gbt_for_rot.cpu().numpy()

        out = {}
        for group in groups:
            bt = int(pose_stack.block_type_ind[group.pose, group.anchor])
            wanted = [
                i
                for i in range(len(pose_of))
                if (int(pose_of[i]), int(block_of[i]), int(bt_of[i]))
                == (group.pose, group.anchor, bt)
            ]
            if not wanted:
                continue
            rows = numpy.nonzero(numpy.isin(gbt_np, wanted))[0]
            if rows.size == 0:
                continue
            out[(group.pose, group.anchor)] = (
                chi_atoms[rows].cpu().numpy(),
                chi[rows].cpu().numpy(),
            )
        return out
