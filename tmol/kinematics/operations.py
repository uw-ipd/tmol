import torch

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor

from tmol.kinematics.compiled import inverse_kin

from .datatypes import KinForest, KinDOF

CoordArray = Tensor[torch.double][:, 3]


@validate_args
def inverseKin(kinforest: KinForest, coords: CoordArray, requires_grad=False) -> KinDOF:
    """xyzs -> HTs, dofs
    - "backward" kinematics
    """
    raw_dofs = inverse_kin(
        coords,
        kinforest.parent,
        kinforest.frame_x,
        kinforest.frame_y,
        kinforest.frame_z,
        kinforest.doftype,
    ).requires_grad_(requires_grad)
    return KinDOF(raw=raw_dofs)
