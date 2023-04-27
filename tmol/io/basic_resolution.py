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

    # step 1: retrieve the global packed_block_types object with the 66 canonical residue types
    # step 2: annotate this PBT with sidechain / backbone kintree data
    # step 3: resolve disulfides
    # step 4: resolve his tautomer
    # step 5: resolve termini variants
    # step 6: assign block-types to each input residue
    # step 7: look for missing atoms
    # step 8: if any atoms missing, build them
    # step 9: construct PoseStack object
    # step 10: copy coordinates into PoseStack tensor

    if atom_is_present is None:
        atom_is_present = torch.all(torch.logical_not(torch.isnan(coords)), dim=3)

    # 1
    # this will return the same object each time to minimize the number
    # of calls to the setup_packed_block_types annotation functions
    pbt = default_canonical_packed_block_types(chain_begin.device)

    # 2
    # TO DO

    # 3
    found_disulfides, res_type_variants = find_disulfides(
        res_types, coords, atom_is_present
    )
    # 4
    his_taut, resolved_coords, resolved_atom_is_present = resolve_his_tautomerization(
        res_types, res_type_variants, coords, atom_is_present
    )

    # 5
    # future!

    # 6
    block_type_assignment = assign_block_types(
        pbt, chain_begin, res_types, restype_variants
    )

    # 7
    missing = note_missing_atoms(
        pbt,
        chain_begin,
        res_types,
        coords,
        atom_is_present,
        found_disulfides,
        his_tautomerization,
    )
