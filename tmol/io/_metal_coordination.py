"""Add or remove one metal-donor bond on an existing pose stack."""

from typing import Optional

import attr
import numpy
import torch

from tmol.io import CanonicalOrdering
from tmol.io._pose_stack_construction import pose_stack_from_canonical_form
from tmol.io._pose_stack_deconstruction import canonical_form_from_pose_stack
from tmol.pose import PoseStack


def add_metal_coordination(
    canonical_ordering: CanonicalOrdering,
    pose_stack: PoseStack,
    pose: int,
    metal: int,
    donor: int,
    atom: str,
    site: Optional[int] = None,
) -> PoseStack:
    """Bond a donor atom to one of a metal's open sites.

    ``site`` defaults to the open site whose virtual points closest to the
    donor, or the first open site of an untemplated ion. Everything else about
    the stack is rebuilt as it was.
    """
    cf = canonical_form_from_pose_stack(canonical_ordering, pose_stack)
    rows = _rows(cf.metal_coordination)
    metal_bt = _block_type(pose_stack, pose, metal)
    if not metal_bt.metal_sites:
        raise ValueError(f"residue {metal} ({metal_bt.name}) is not a metal")
    filled = {site for p, m, site, _, _ in rows if (p, m) == (pose, metal)}
    open_sites = [
        k
        for k in range(len(metal_bt.metal_sites[0].site_connections))
        if k not in filled
    ]
    if site is None:
        if not open_sites:
            raise ValueError(f"{metal_bt.name} at residue {metal} has no open site")
        site = _closest_open_site(pose_stack, pose, metal, donor, atom, open_sites)
    elif site not in open_sites:
        raise ValueError(f"site {site} of {metal_bt.name} is not open")

    donor_bt = _block_type(pose_stack, pose, donor)
    index = canonical_ordering.restypes_atom_index_mapping[donor_bt.io_equiv_class]
    rows.append((pose, metal, site, donor, index[atom]))
    return _rebuild(canonical_ordering, pose_stack, cf, rows)


def remove_metal_coordination(
    canonical_ordering: CanonicalOrdering,
    pose_stack: PoseStack,
    pose: int,
    metal: int,
    site: int,
) -> PoseStack:
    """Open one filled site of a metal; the donor returns to its plain form."""
    cf = canonical_form_from_pose_stack(canonical_ordering, pose_stack)
    rows = _rows(cf.metal_coordination)
    kept = [r for r in rows if r[:3] != (pose, metal, site)]
    if len(kept) == len(rows):
        raise ValueError(f"site {site} of residue {metal} is not filled")
    return _rebuild(canonical_ordering, pose_stack, cf, kept)


def _rows(metal_coordination):
    if metal_coordination is None:
        return []
    return [tuple(r) for r in metal_coordination.tolist()]


def _block_type(pose_stack, pose, res):
    bt_ind = int(pose_stack.block_type_ind64[pose, res])
    if bt_ind < 0:
        raise ValueError(f"pose {pose} has no residue {res}")
    return pose_stack.packed_block_types.active_block_types[bt_ind]


def _closest_open_site(pose_stack, pose, metal, donor, atom, open_sites):
    metal_bt = _block_type(pose_stack, pose, metal)
    site = metal_bt.metal_sites[0]
    if not site.site_virts:
        return open_sites[0]
    xyz = pose_stack.coords[pose].detach().cpu().numpy().astype(numpy.float64)
    offset = int(pose_stack.block_coord_offset64[pose, metal])
    metal_xyz = xyz[offset + metal_bt.atom_to_idx[site.metal_atom]]
    donor_bt = _block_type(pose_stack, pose, donor)
    donor_offset = int(pose_stack.block_coord_offset64[pose, donor])
    direction = xyz[donor_offset + donor_bt.atom_to_idx[atom]] - metal_xyz

    def alignment(k):
        ray = xyz[offset + metal_bt.atom_to_idx[site.site_virts[k]]] - metal_xyz
        return numpy.dot(ray, direction) / numpy.linalg.norm(ray)

    return max(open_sites, key=alignment)


def _rebuild(canonical_ordering, pose_stack, cf, rows):
    device = pose_stack.device
    coordination = (
        torch.tensor(rows, dtype=torch.int64, device=device).view(-1, 5)
        if rows
        else None
    )
    cf = attr.evolve(cf, metal_coordination=coordination)
    return pose_stack_from_canonical_form(
        canonical_ordering,
        pose_stack.packed_block_types,
        *cf,
        trust_hydrogen_names=True,
        find_additional_disulfides=False,
        find_additional_cyclic_closures=False,
        find_additional_metal_coordination=False,
    )
