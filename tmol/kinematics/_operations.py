import torch

from tmol.types._functional import _validate_args
from tmol.types._torch import Tensor
from ._datatypes import KinForest, KinDOF

CoordArray = Tensor[torch.double][:, 3]


@_validate_args
def inverseKin(kinforest: KinForest, coords: CoordArray, requires_grad=False) -> KinDOF:
    """xyzs -> HTs, dofs
    - "backward" kinematics
    """
    from tmol.kinematics.compiled._compiled_ops import _inverse_kin

    raw_dofs = _inverse_kin(
        coords,
        kinforest.parent,
        kinforest.frame_x,
        kinforest.frame_y,
        kinforest.frame_z,
        kinforest.doftype,
    ).requires_grad_(requires_grad)
    return KinDOF(raw=raw_dofs)
