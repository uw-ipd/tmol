import torch
from tmol.types.torch import Tensor
from typing import Optional

from tmol.pose.pose_stack import PoseStack

from tmol.io.details.canonical_packed_block_types import (
    default_canonical_packed_block_types
)
from tmol.io.details.disulfide_search import find_disulfides
from tmol.io.details.his_taut_resolution import resolve_his_tautomerization


def pose_stack_from_canonical_form(
    chain_begin: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Optional[Tensor[torch.int32][:, :, :]] = None,
) -> PoseStack:
    """"Create a PoseStack given atom coordinates in canonical ordering"""

    assert chain_begin.device == res_types.device
    assert chain_begin.device == coords.device

    # step 1: retrieve the global packed_block_types object with the 66
    #         canonical residue types
    # step 2: resolve disulfides
    # step 3: resolve his tautomer
    # step 4: resolve termini variants
    # step 5: assign block-types to each input residue
    # step 6: select the atoms from the canonically-ordered input tensors
    #         (the coords and atom_is_present tensors) that belong to the
    #         now-assigned block types, discarding/ignoring
    #         any others that may have been provided
    # step 7: if any atoms missing, build them
    # step 8: construct PoseStack object
    # step 9: copy coordinates into PoseStack tensor

    if atom_is_present is None:
        atom_is_present = torch.all(torch.logical_not(torch.isnan(coords)), dim=3)

    # 1
    # this will return the same object each time to minimize the number
    # of calls to the setup_packed_block_types annotation functions
    pbt = default_canonical_packed_block_types(chain_begin.device)

    # 2
    found_disulfides, res_type_variants = find_disulfides(
        res_types, coords, atom_is_present
    )
    # 3
    his_taut, resolved_coords, resolved_atom_is_present = resolve_his_tautomerization(
        res_types, res_type_variants, coords, atom_is_present
    )

    # 4
    # future!

    # 5
    block_types = assign_block_types(pbt, chain_begin, res_types, restype_variants)

    # 6
    block_type_coords, missing_atoms = take_block_type_atoms_from_canonical(
        pbt,
        chain_begin,
        block_types,
        coords,
        atom_is_present,
        found_disulfides,
        his_tautomerization,
    )
