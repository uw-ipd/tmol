import attr
import numpy

import torch

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor
from tmol.types.attrs import ValidateAttrs

from tmol.kinematics.compiled import compiled

from .datatypes import NodeType, KinTree, KinDOF
from .gpu_operations import GPUKinTreeReordering
from .cpu_operations import iterative_f1f2_summation

from .scan_ordering import KinTreeScanOrdering

HTArray = Tensor(torch.double)[:, 4, 4]
CoordArray = Tensor(torch.double)[:, 3]


@attr.s(frozen=True, auto_attribs=True, slots=True)
class BackKinResult(ValidateAttrs):
    hts: HTArray
    dofs: KinDOF


@validate_args
def backwardKin(kintree: KinTree, coords: CoordArray) -> BackKinResult:
    """xyzs -> HTs, dofs

      - "backward" kinematics
    """
    natoms = coords.shape[0]

    # 1) global HTs
    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0

    dofs = KinDOF.full(natoms, numpy.nan, device=coords.device)
    HTs = compiled.backward_kin(
        coords,
        kintree.doftype,
        kintree.parent,
        kintree.frame_x,
        kintree.frame_y,
        kintree.frame_z,
        dofs.raw,
    )

    return BackKinResult(HTs, dofs)


def DOFTransforms(dof_types: Tensor(torch.int)[:], dofs: KinDOF) -> HTArray:
    """Calculate HT representation of given dofs."""
    return compiled.dof_transforms(dofs.raw, dof_types)


@attr.s(frozen=True, auto_attribs=True, slots=True)
class ForwardKinResult:
    hts: HTArray
    coords: CoordArray


@validate_args
def forwardKin(kintree: KinTree, dofs: KinDOF) -> ForwardKinResult:
    """dofs -> HTs, xyzs

      - "forward" kinematics
    """
    assert len(kintree) == len(dofs)
    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0

    # 1) Calculate local inter-node HTs from dofs
    ordering = KinTreeScanOrdering.for_kintree(kintree)

    HTs = compiled.forward_kin(
        dofs.raw, kintree.doftype, **attr.asdict(ordering.forward_scan_paths)
    )

    coords = HTs[:, :3, 3]
    return ForwardKinResult(HTs, coords)


@validate_args
def resolveDerivs(
    kintree: KinTree, dofs: KinDOF, HTs: HTArray, dsc_dx: CoordArray
) -> KinDOF:
    """xyz derivs -> dof derivs

    - derivative mapping using Abe and Go approach
    """

    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0

    assert len(kintree) == len(dofs)
    assert len(kintree) == len(HTs)
    assert len(kintree) == len(dsc_dx)

    # Can not render higher-order derivatives for kinematic operations,
    # as is generated with backward(create_graph=True) is called.
    assert dsc_dx.requires_grad is False

    # 1) local f1/f2s
    Xs = HTs[:, 0:3, 3]

    f1s = torch.cross(Xs, Xs - dsc_dx)
    f2s = dsc_dx.clone()  # clone input buffer before aggregation

    # 2) pass f1/f2s up tree
    f1f2s = torch.cat((f1s, f2s), 1)

    if f1f2s.device.type == "cuda":
        (
            GPUKinTreeReordering.for_kintree(kintree).derivsum_ordering.segscan_f1f2s(
                f1f2s, inplace=True
            )
        )

    else:
        iterative_f1f2_summation(f1f2s, kintree.parent, inplace=True)

    # 3) convert to dscore/dtors
    dsc_ddofs = compiled.f1f2_to_deriv(
        HTs, dofs.raw, kintree.doftype, kintree.parent, f1f2s
    )

    return KinDOF(raw=dsc_ddofs)
