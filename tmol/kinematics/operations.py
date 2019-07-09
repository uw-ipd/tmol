import torch

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor

from tmol.kinematics.compiled import compiled
from tmol.kinematics.script_modules import KinematicModule


from .datatypes import KinTree, KinDOF

CoordArray = Tensor(torch.double)[:, 3]


@validate_args
def inverseKin(kintree: KinTree, coords: CoordArray, requires_grad=False) -> KinDOF:
    """xyzs -> HTs, dofs
      - "backward" kinematics
    """
    raw_dofs = compiled.inverse_kin(
        coords,
        kintree.parent,
        kintree.frame_x,
        kintree.frame_y,
        kintree.frame_z,
        kintree.doftype,
    ).requires_grad_(requires_grad)
    return KinDOF(raw=raw_dofs)


@validate_args
def forwardKin(kintree: KinTree, dofs: KinDOF) -> CoordArray:
    """dofs -> HTs, xyzs
      - "forward" kinematics
    """

    ksm = KinematicModule(kintree, dofs.raw.device)
    coords = ksm(dofs.raw)

    return coords
