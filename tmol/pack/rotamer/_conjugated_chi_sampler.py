"""Rotamers for the blocks bonded to an amino acid's sidechain.

A glycan or a ligand joined to a sidechain cannot be sampled a residue at a
time: the torsion about a linkage bond has its fourth atom in the neighbouring
residue, so in a single-block kinforest it has nothing to turn. The group --
anchor plus everything bonded to it -- is enumerated as one unit instead, with
the anchor's own chi coming from its rotamer library and the attached blocks'
chi from their chi_samples.

This sampler owns only the latter. It never claims a block that Dunbrack or the
nucleic-acid sampler covers, and it never claims a free ligand: a residue has to
be bonded to a sidechain to be here at all.
"""

import attr
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
from tmol.kinematics import KinForest
from tmol.pack.rotamer._chi_sampler import ChiSampler
from tmol.pack.rotamer._conformer_sampler import sc_roots_for_chis
from tmol.pack.rotamer._conjugated_groups import (
    find_conjugated_groups,
    group_sampled_chi,
)
from tmol.pack._packer_task import (
    DEFAULT_CHI_SAMPLE_EXPANDED_LIMIT,
    DEFAULT_CHI_SAMPLE_LIMIT,
)


def _heavy_chi(rt: RefinedResidueType):
    """The chi this sampler is responsible for: sampled, and not a proton chi."""
    return [cs for cs in rt.chi_samples if not cs.is_proton]


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
        return len(_heavy_chi(rt)) > 0

    def defines_rotamers_for_bts(
        self, pbt: PackedBlockTypes, bt_inds: Tensor[torch.int64]
    ) -> Tensor[torch.bool]:
        builds = torch.tensor(
            [len(_heavy_chi(bt)) > 0 for bt in pbt.active_block_types],
            dtype=torch.bool,
            device=pbt.device,
        )
        return builds[bt_inds]

    @validate_args
    def first_sc_atoms_for_rt(self, rt: RefinedResidueType) -> Tuple[str, ...]:
        return sc_roots_for_chis(rt, [cs.chi_dihedral for cs in _heavy_chi(rt)])

    def group_conformers(self, pose_stack: PoseStack, anchor_chi=None, budget=None):
        """Enumerate each group's product using actual library cardinality.

        Limits count all member rotamers, including the offered input state.
        Library states are retained; an impossible limit fails before allocating
        the Cartesian product. Frozen child chi retain their input geometry.
        """
        expanded_limit, limit = budget or (
            self.chi_sample_expanded_limit,
            self.chi_sample_limit,
        )
        out = []
        pbt = pose_stack.packed_block_types
        for group in find_conjugated_groups(pose_stack):
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
                    if name.startswith("chi")
                }
                keep_cols = []
                for j in range(anchor_atoms.shape[1]):
                    atom = int(anchor_atoms[0, j])
                    if atom < 0 or atom not in atom_to_chi:
                        continue
                    name, ab, ac = atom_to_chi[atom]
                    anchor_cols.append((0, name, ab, ac))
                    keep_cols.append(j)
                if anchor_cols:
                    lib = anchor_values[:, keep_cols]

            kept = group_sampled_chi(
                group,
                pose_stack,
                expanded_limit,
                limit,
                library_size=lib.shape[0],
                reserve_current=self.include_current,
            )
            columns, per_chi = [], []
            for owner, cs in kept:
                bt = pbt.active_block_types[
                    int(pose_stack.block_type_ind[group.pose, group.blocks[owner]])
                ]
                uaids = bt.torsion_to_uaids[cs.chi_dihedral]
                columns.append((owner, cs.chi_dihedral, uaids[1][0], uaids[2][0]))
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
        if not self.include_current:
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
            int(pose_stack.block_type_ind[group.pose, b]) for b in group.blocks
        )
        key = (types, group.links)
        cache = getattr(pbt, "conjugated_kinforest_cache", None)
        if cache is None:
            cache = {}
            setattr(pbt, "conjugated_kinforest_cache", cache)
        if key not in cache:
            cache[key] = construct_block_group_kinforest(
                [pbt.active_block_types[t] for t in types], group.links, anchor=0
            )
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
            nb = int(kfo_for_atom[int(offsets[owner]) + int(atom_b)])
            nc = int(kfo_for_atom[int(offsets[owner]) + int(atom_c)])
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

        Every member of a group gets the same number of rotamers, and rotamer k
        of each member belongs to group conformer k. That correspondence is what
        keeps them in step; it is recorded so the packer can later treat a
        group's rotamers as one choice rather than several.
        """
        device = pose_stack.device
        n_gbt = task.cons_bt_pose.shape[0]
        n_rots_for_gbt = torch.zeros(n_gbt, dtype=torch.int32, device=device)

        pose_of = task.cons_bt_pose.cpu().numpy()
        block_of = task.cons_bt_block.cpu().numpy()
        bt_of = task.cons_bt_block_type.cpu().numpy()
        gbt_for = {}
        for i in range(n_gbt):
            gbt_for[(int(pose_of[i]), int(block_of[i]), int(bt_of[i]))] = i

        anchor_chi = self.anchor_library_chi(
            pose_stack, task, find_conjugated_groups(pose_stack)
        )
        groups = self.group_conformers(
            pose_stack, anchor_chi, budget=getattr(task, "chi_sample_budget", None)
        )
        emitted = []
        for gi, (group, columns, conformers) in enumerate(groups):
            n_conf = int(conformers.shape[0])
            # the anchor is emitted too when its own chi are part of the
            #    product; otherwise it keeps the conformation it came in with
            first_owner = 0 if any(c[0] == 0 for c in columns) else 1
            for owner in range(first_owner, len(group)):
                block = group.blocks[owner]
                bt = int(pose_stack.block_type_ind[group.pose, block])
                gbt = gbt_for.get((group.pose, block, bt))
                if gbt is None:
                    continue  # the packer is not considering this block type
                n_rots_for_gbt[gbt] = n_conf
                emitted.append((gbt, gi, owner, n_conf))

        # rotamers must be listed in ascending block-type order: the merge
        #    locates a sampler's rotamers by a cumulative sum over the counts
        #    above, so any other order silently points them at the wrong
        #    conformers
        emitted.sort()
        gbt_for_rotamer, plan = [], []
        for gbt, gi, owner, n_conf in emitted:
            plan.append((gi, owner, gbt, len(gbt_for_rotamer)))
            gbt_for_rotamer.extend([gbt] * n_conf)

        return (
            n_rots_for_gbt,
            torch.tensor(gbt_for_rotamer, dtype=torch.int32, device=device),
            dict(groups=groups, plan=plan),
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
        from tmol.kinematics.compiled import inverse_kin, forward_only_op
        from tmol.pack.rotamer._single_residue_kinforest import (
            construct_single_residue_kinforest,
        )

        pbt = pose_stack.packed_block_types
        device = pose_stack.device

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(x, d=device):
            return torch.tensor(numpy.asarray(x, dtype=numpy.int32), device=d)

        for gi, (group, columns, conformers) in enumerate(sample_dict["groups"]):
            kinforest, dofs, offsets = self.group_coords(
                pose_stack, group, columns, conformers
            )
            rot_kf, _ = self._group_kinforest(pose_stack, group)
            stack = _p(
                torch.stack(
                    [
                        kinforest.id,
                        kinforest.doftype,
                        kinforest.parent,
                        kinforest.frame_x,
                        kinforest.frame_y,
                        kinforest.frame_z,
                    ],
                    dim=1,
                )
            )
            n_group_atoms = int(offsets[-1])
            nodes = _p(_t(rot_kf.nodes))
            scans = _p(_t(rot_kf.scans))
            gens = _p(_t(rot_kf.gens, torch.device("cpu")))
            atom_order = kinforest.id[1:].to(torch.int64)
            folded = []
            for i in range(dofs.shape[0]):
                kco = forward_only_op(
                    dofs[i],
                    nodes,
                    scans,
                    gens,
                    stack,
                )
                rto = torch.zeros(
                    (n_group_atoms, 3), dtype=torch.float32, device=device
                )
                rto[atom_order] = kco[1:]
                folded.append(rto)

            for pgi, owner, gbt, first in sample_dict["plan"]:
                if pgi != gi:
                    continue
                block = group.blocks[owner]
                bt = pbt.active_block_types[
                    int(pose_stack.block_type_ind[group.pose, block])
                ]
                construct_single_residue_kinforest(bt)
                mkf = bt.rotamer_kinforest
                mk = KinForest(
                    id=_t(numpy.concatenate([[-1], mkf.id])),
                    doftype=_t(numpy.concatenate([[0], mkf.doftype])),
                    parent=_t(numpy.concatenate([[0], mkf.parent + 1])),
                    frame_x=_t(numpy.concatenate([[0], mkf.frame_x + 1])),
                    frame_y=_t(numpy.concatenate([[0], mkf.frame_y + 1])),
                    frame_z=_t(numpy.concatenate([[0], mkf.frame_z + 1])),
                )
                lo, hi = int(offsets[owner]), int(offsets[owner + 1])
                for k in range(len(folded)):
                    member = folded[k][lo:hi]
                    kco = torch.cat(
                        [
                            torch.zeros((1, 3), dtype=torch.float32, device=device),
                            member,
                        ]
                    )[mk.id.to(torch.int64) + 1]
                    kco[0, :] = 0
                    member_dofs = inverse_kin(
                        kco, mk.parent, mk.frame_x, mk.frame_y, mk.frame_z, mk.doftype
                    )
                    conf = int(conf_inds_for_sampler[first + k])
                    off = int(n_dof_atoms_offset_for_conformer[conf]) + 1
                    n_member = hi - lo
                    if off + n_member > conf_dofs_kto.shape[0]:
                        raise IndexError(
                            f"conjugated rotamer {first + k} (group block "
                            f"{block}, conformer {k}) maps to global conformer "
                            f"{conf}, whose dof offset {off} leaves only "
                            f"{conf_dofs_kto.shape[0] - off} rows for "
                            f"{n_member} atoms; sampler rotamers "
                            f"{conf_inds_for_sampler.shape[0]}, total dof rows "
                            f"{conf_dofs_kto.shape[0]}"
                        )
                    conf_dofs_kto[off : off + n_member, :] = member_dofs[1:]

    def anchor_library_chi(self, pose_stack, task, groups):
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
        allowed = torch.zeros_like(task.per_block_conformer_sampler_allowed)
        library_task.per_block_conformer_sampler_allowed = allowed
        for group in groups:
            allowed[group.pose, group.anchor, index] = True
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
