"""Handle degenerate deposited bond geometry before packing or scoring."""

import copy
import warnings

import torch

from tmol.pose import PoseStack
from tmol.io.details._build_missing_leaf_atoms import _build_coords_from_icoors


def rebuild_coincident_hydrogens(pose: PoseStack) -> PoseStack:
    """Rebuild hydrogens on their bonded parent; reject coincident heavy bonds.

    Chemistry has already been selected, so rebuilding coordinates preserves
    protonation. Virtual sites are excluded. A bond shorter than 1e-4 Angstrom
    is numerically degenerate, far below the precision of deposited coordinates.
    Other bond-length and angle outliers are left to the force field.
    """
    pbt = pose.packed_block_types
    types = pose.block_type_ind64
    positions, blocks, bonds = torch.nonzero(
        (types >= 0).unsqueeze(-1) & pbt.bond_is_real[types.clamp_min(0)],
        as_tuple=True,
    )
    atoms = pbt.bond_indices[types[positions, blocks], bonds].long()
    first = torch.column_stack((positions, blocks, atoms[:, 0]))
    second = torch.column_stack((positions, blocks, atoms[:, 1]))

    positions, blocks, connections = torch.nonzero(
        pose.inter_residue_connections64[..., 0] >= 0, as_tuple=True
    )
    partner, partner_connection = pose.inter_residue_connections64[
        positions, blocks, connections
    ].T
    source_atoms = pbt.conn_atom[types[positions, blocks], connections]
    partner_atoms = pbt.conn_atom[types[positions, partner], partner_connection]
    first = torch.cat((first, torch.column_stack((positions, blocks, source_atoms))))
    second = torch.cat(
        (second, torch.column_stack((positions, partner, partner_atoms)))
    )

    def coordinates(endpoint):
        pi, bi, ai = endpoint.T
        return pose.coords[pi, pose.block_coord_offset64[pi, bi] + ai]

    coincident = ((coordinates(first) - coordinates(second)) ** 2).sum(-1) < 1e-8
    if not coincident.any():
        return pose
    # Inspect only degenerate bonds on CPU; ordinary structures stay on device.
    element = {atom.name: atom.element.upper() for atom in pbt.chem_db.atom_types}
    repair, invalid = set(), []

    def identify(endpoint):
        pi, bi, ai = endpoint
        residue = pbt.active_block_types[int(types[pi, bi])]
        atom = residue.atoms[ai]
        label = f"pose {pi}, residue {bi} {residue.name}, atom {atom.name}"
        if pose.pdb_info is not None and pose.pdb_info.residue_labels is not None:
            label += (
                f" (chain {pose.pdb_info.chain_labels[pi, bi]}, "
                f"resid {pose.pdb_info.residue_labels[pi, bi]})"
            )
        return element[atom.atom_type], label

    for left, right in zip(first[coincident].tolist(), second[coincident].tolist()):
        if tuple(left) >= tuple(right):
            continue
        elements, labels = zip(identify(left), identify(right))
        if "VR" in elements:
            continue
        hydrogens = [
            tuple(end)
            for end, elem in zip((left, right), elements)
            if elem in ("H", "D")
        ]
        if hydrogens:
            repair.update(hydrogens)
        else:
            invalid.append(" -- ".join(labels))
    if invalid:
        raise ValueError(
            "Coincident bonded heavy atoms in input geometry: "
            + "; ".join(invalid[:20])
        )
    if not repair:
        return pose
    warnings.warn(
        "Rebuilding hydrogens coincident with their bonded parent: "
        + "; ".join(identify(end)[1] for end in sorted(repair)[:20]),
        stacklevel=2,
    )
    pi, bi, ai = torch.tensor(sorted(repair), device=pose.device).T
    targets = torch.zeros(
        (*types.shape, pbt.max_n_atoms), dtype=torch.bool, device=pose.device
    )
    targets[pi, bi, ai] = True
    coords = pose.coords.clone()
    coords[pi, pose.block_coord_offset64[pi, bi] + ai] = torch.nan
    result = copy.copy(pose)
    result.coords = _build_coords_from_icoors(
        pbt,
        coords,
        targets,
        torch.isnan(coords).any(-1),
        pose.block_coord_offset,
        pose.block_type_ind,
        pose.inter_residue_connections,
    )
    return result
