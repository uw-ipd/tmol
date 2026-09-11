"""Resolve explicit and geometrically inferred head-to-tail polymer closures."""

import torch


def find_cyclic_closures(
    canonical_ordering,
    chain_id,
    res_types,
    coords,
    cyclic_bonds=None,
    find_additional_cyclic_closures=True,
    cutoff_dis=2.0,
):
    """Return [pose, up residue, down residue] rows on the input device.

    Infer only closures between the ends of the same contiguous polymer run.
    Missing connection atoms, padding, single residues and nonpolymers cannot
    create a closure. Explicit rows take precedence over geometric inference.
    """
    device = res_types.device
    explicit = (
        torch.empty((0, 3), dtype=torch.int64, device=device)
        if cyclic_bonds is None
        else cyclic_bonds
    )
    if not find_additional_cyclic_closures or res_types.shape[1] == 0:
        return explicit

    connection = canonical_ordering.polymer_conn_inds
    up = torch.as_tensor(connection.up_atom_for_co_restype, device=device)
    down = torch.as_tensor(connection.down_atom_for_co_restype, device=device)
    if up.numel() == 0:
        return explicit
    safe_types = res_types.clamp_min(0).long()
    up_atoms, down_atoms = up[safe_types], down[safe_types]
    polymer = (res_types >= 0) & (chain_id >= 0) & (
        (up_atoms >= 0) | (down_atoms >= 0)
    )
    adjacent = (
        polymer[:, :-1]
        & polymer[:, 1:]
        & (chain_id[:, :-1] == chain_id[:, 1:])
    )
    starts = polymer.clone()
    ends = polymer.clone()
    starts[:, 1:] &= ~adjacent
    ends[:, :-1] &= ~adjacent
    pose, first = starts.nonzero(as_tuple=True)
    end_pose, last = ends.nonzero(as_tuple=True)
    # Starts and ends enumerate the same runs in row-major order.
    valid = (pose == end_pose) & (first != last)
    ua, da = up_atoms[pose, last], down_atoms[pose, first]
    valid &= (ua >= 0) & (da >= 0)
    delta = coords[pose, last, ua.clamp_min(0)] - coords[
        pose, first, da.clamp_min(0)
    ]
    distance2 = delta.square().sum(-1)
    valid &= torch.isfinite(distance2) & (distance2 < cutoff_dis**2)
    if explicit.shape[0]:
        used_up = torch.zeros_like(polymer)
        used_down = torch.zeros_like(polymer)
        used_up[explicit[:, 0], explicit[:, 1]] = True
        used_down[explicit[:, 0], explicit[:, 2]] = True
        valid &= ~used_up[pose, last] & ~used_down[pose, first]
    inferred = torch.stack((pose[valid], last[valid], first[valid]), dim=1)
    return torch.cat((explicit, inferred), dim=0)
