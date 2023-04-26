import torch
from tmol.types.torch import Tensor
from typing import Optional

from tmol.pose.pose_stack import PoseStack


def pose_stack_from_canonical_form(
    chain_begin: Tensor[torch.int32][:, :],
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Optional[Tensor[torch.int32][:, :, :]],
) -> PoseStack:
    """"Create a PoseStack given atom coordinates in canonical ordering"""

    # step 1: retrieve the global packed_block_types object with the 66 canonical residue types
    # step 2: annotate this PBT with sidechain / backbone kintree data
    # step 3: resolve disulfides
    # step 4: resolve his tautomer
    # step 5: resolve termini variants
    # step 6: look for missing atoms
    # step 7: if any atoms missing, build them
    # step 8: construct PoseStack object
    # step 9: copy coordinates into PoseStack tensor
