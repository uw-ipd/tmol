import torch
from tmol.types.torch import Tensor
from typing import Optional

from tmol.pose.pose_stack import PoseStack

from tmol.io.details.canonical_packed_block_types import (
    default_canonical_packed_block_types,
)
from tmol.io.details.disulfide_search import find_disulfides
from tmol.io.details.his_taut_resolution import resolve_his_tautomerization
from tmol.io.details.select_from_canonical import (
    assign_block_types,
    take_block_type_atoms_from_canonical,
)


def pose_stack_from_canonical_form(
    chain_begin: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Optional[Tensor[torch.int32][:, :, :]] = None,
) -> PoseStack:
    """ "Create a PoseStack given atom coordinates in canonical ordering"""

    assert chain_begin.device == res_types.device
    assert chain_begin.device == coords.device

    # step 1: retrieve the global packed_block_types object with the 66
    #         canonical residue types
    # step 2: resolve disulfides
    # step 3: resolve his tautomer
    # step 4: resolve termini variants (future)
    # step 5: assign block-types to each input residue
    # step 6: select the atoms from the canonically-ordered input tensors
    #         (the coords and atom_is_present tensors) that belong to the
    #         now-assigned block types, discarding/ignoring
    #         any others that may have been provided
    # step 7: if any atoms missing, build them
    # step 8: construct PoseStack object

    if atom_is_present is None:
        atom_is_present = torch.all(torch.logical_not(torch.isnan(coords)), dim=3)

    # 1
    # this will return the same object each time to minimize the number
    # of calls to the setup_packed_block_types annotation functions
    pbt, atom_type_resolver = default_canonical_packed_block_types(chain_begin.device)

    # 2
    found_disulfides, res_type_variants = find_disulfides(
        res_types, coords, atom_is_present
    )
    # 3
    (
        his_taut,
        res_type_variants,
        resolved_coords,
        resolved_atom_is_present,
    ) = resolve_his_tautomerization(
        res_types, res_type_variants, coords, atom_is_present
    )

    # 4
    # future!

    # 5
    (
        block_types64,
        inter_residue_connections64,
        inter_block_bondsep64,
    ) = assign_block_types(
        pbt, chain_begin, res_types, res_type_variants, found_disulfides
    )

    # 6
    block_coords, missing_atoms, real_atoms = take_block_type_atoms_from_canonical(
        pbt,
        chain_begin,
        block_types64,
        coords,
        atom_is_present,
        found_disulfides,
        his_tautomerization,
    )

    # 7
    pose_stack_coords, block_coord_offset = build_missing_hydrogens(
        pbt,
        atom_type_resolver,
        block_types64,
        real_atoms,
        block_type_coords,
        missing_atoms,
    )

    def i64(x):
        return x.to(torch.int64)

    def i32(x):
        return x.to(torch.int32)

    # 8
    return PoseStack(
        packed_block_types=pbt,
        coords=pose_stack_coords,
        block_coord_offset=block_coord_offset,
        block_coord_offset64=i64(block_coord_offset),
        inter_residue_connections=i32(inter_residue_connections64),
        inter_residue_connections64=inter_residue_connections64,
        inter_block_bondsep=i32(inter_block_bondsep64),
        inter_block_bondsep64=inter_block_bondsep64,
        block_type_ind=i32(block_types64),
        block_type_ind64=block_types64,
        device=pbt.device,
    )
