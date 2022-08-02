import torch

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor

from tmol.kinematics.compiled import inverse_kin
from tmol.kinematics.script_modules import KinematicModule


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


# wrapper for the module that is used for testing
# creates a new module each call so not particularly efficient
@validate_args
def forwardKin(kinforest: KinForest, dofs: KinDOF) -> CoordArray:
    """dofs -> HTs, xyzs
    - "forward" kinematics
    """

    ksm = KinematicModule(kinforest, dofs.raw.device)
    coords = ksm(dofs.raw)

    return coords
