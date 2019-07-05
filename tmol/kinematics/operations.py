import attr
import numpy

import torch

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor
from tmol.types.attrs import ValidateAttrs

from tmol.kinematics.compiled import compiled
from tmol.kinematics.script_modules import KinematicScoreModule


from .datatypes import NodeType, KinTree, KinDOF

from .scan_ordering import KinTreeScanOrdering

HTArray = Tensor(torch.double)[:, 4, 4]
CoordArray = Tensor(torch.double)[:, 3]


@validate_args
def inverseKin(kintree: KinTree, coords: CoordArray) -> BackKinResult:
    """xyzs -> HTs, dofs
      - "backward" kinematics
    """
    natoms = coords.shape[0]
    dofs = KinDOF.full(natoms, numpy.nan, device=coords.device)
    dofs.raw = compiled.inverse_kin(coords, kintree)

    return dofs


@validate_args
def forwardKin(kintree: KinTree, dofs: KinDOF) -> ForwardKinResult:
    """dofs -> HTs, xyzs
      - "forward" kinematics
    """

    ksm = KinematicScoreModule(kintree)
    coords = ksm(dofs)

    return coords
