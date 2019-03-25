import attr
import numpy

import torch
from typing import Optional

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor
from tmol.types.attrs import ValidateAttrs

from tmol.kinematics.compiled import compiled

from .datatypes import NodeType, KinTree, KinDOF, BondDOF, JumpDOF
from .gpu_operations import GPUKinTreeReordering
from .cpu_operations import iterative_f1f2_summation

from .scan_ordering import KinTreeScanOrdering

HTArray = Tensor(torch.double)[:, 4, 4]
CoordArray = Tensor(torch.double)[:, 3]


@validate_args
def JumpDerivatives(
    dofs: JumpDOF, Ms: HTArray, Mparents: HTArray, f1s: CoordArray, f2s: CoordArray
) -> JumpDOF:
    """
    compute JUMP derivatives from f1/f2

    Translational derivatives are straightforward dot products of f2s
        (the downstream derivative sum)

    Rotational derivatives use the Abe and Go "trick" that allows us to easily
    compute derivatives with respect to rotation about an axis.

    In this case, there are three axes to compute derivatives of::
        1) the Z axis (alpha rotation)
        2) the Y axis after applying the alpha rotation (beta rotation)
        3) the X axis after applying the alpha & beta rot (gamma rotation)

    Derivatives are ONLY assigned to the RBdel DOFs
    """
    # trans dofs
    njumpatoms, = dofs.shape
    dsc_ddofs = JumpDOF.zeros((njumpatoms,), device=dofs.raw.device)

    x_axes = Mparents[:, 0:3, 0]
    y_axes = Mparents[:, 0:3, 1]
    z_axes = Mparents[:, 0:3, 2]

    # einsums here are taking dot products of the vector stacks
    dsc_ddofs.RBx[:] = torch.einsum("ij,ij->i", (x_axes, f2s))
    dsc_ddofs.RBy[:] = torch.einsum("ij,ij->i", (y_axes, f2s))
    dsc_ddofs.RBz[:] = torch.einsum("ij,ij->i", (z_axes, f2s))

    end_pos = Ms[:, 0:3, 3]
    rotdof3_axes = -Mparents[:, 0:3, 2]

    zrots = torch.zeros([njumpatoms, 3, 3], dtype=torch.double, device=dofs.raw.device)
    zrots[:, 0, 0] = torch.cos(dofs.RBdel_gamma)
    zrots[:, 0, 1] = -torch.sin(dofs.RBdel_gamma)
    zrots[:, 1, 0] = torch.sin(dofs.RBdel_gamma)
    zrots[:, 1, 1] = torch.cos(dofs.RBdel_gamma)
    zrots[:, 2, 2] = 1
    rotdof2_axes = -torch.matmul(Mparents[:, 0:3, 0:3], zrots)[:, 0:3, 1]

    yrots = torch.zeros([njumpatoms, 3, 3], dtype=torch.double, device=dofs.raw.device)
    yrots[:, 0, 0] = torch.cos(-dofs.RBdel_beta)
    yrots[:, 0, 2] = -torch.sin(-dofs.RBdel_beta)
    yrots[:, 1, 1] = 1
    yrots[:, 2, 0] = torch.sin(-dofs.RBdel_beta)
    yrots[:, 2, 2] = torch.cos(-dofs.RBdel_beta)
    rotdof1_axes = -torch.matmul(torch.matmul(Mparents[:, 0:3, 0:3], zrots), yrots)[
        :, 0:3, 0
    ]

    # einsums here are taking dot products of the vector stacks
    dsc_ddofs.RBdel_alpha[:] = torch.einsum(
        "ij,ij->i", (rotdof1_axes, f1s)
    ) + torch.einsum("ij,ij->i", (torch.cross(rotdof1_axes, end_pos), f2s))
    dsc_ddofs.RBdel_beta[:] = torch.einsum(
        "ij,ij->i", (rotdof2_axes, f1s)
    ) + torch.einsum("ij,ij->i", (torch.cross(rotdof2_axes, end_pos), f2s))
    dsc_ddofs.RBdel_gamma[:] = torch.einsum(
        "ij,ij->i", (rotdof3_axes, f1s)
    ) + torch.einsum("ij,ij->i", (torch.cross(rotdof3_axes, end_pos), f2s))

    return dsc_ddofs


@validate_args
def BondDerivatives(
    dofs: BondDOF, Ms: HTArray, Mparents: HTArray, f1s: CoordArray, f2s: CoordArray
) -> BondDOF:
    """
    compute JUMP derivatives from f1/f2

    The d derivatives are straightforward dot products of f2s
        (the downstream derivative sum)

    Other DOF derivatives use the Abe and Go "trick" that allows us to
        easily compute derivatives with respect to rotation about an axis.
    The phi_p and phi_c derivs are simply rotation about the X axis of the
        parent and child coordinate frame, respectively
    The theta derivs are more complex. Similar to jump derivs, we need to
        UNDO the phi_c rotation, and then take the Z axis of the child HT
    """

    nbondatoms, = dofs.shape

    end_p_pos = Mparents[:, 0:3, 3]
    phi_p_axes = Mparents[:, 0:3, 0]
    end_c_pos = Ms[:, 0:3, 3]
    phi_c_axes = Ms[:, 0:3, 0]

    # to get the theta axis, we need to undo the phi_c rotation (about x)
    phicrots = Ms.new_zeros([nbondatoms, 3, 3])
    phicrots[:, 0, 0] = 1
    phicrots[:, 1, 1] = torch.cos(-dofs.phi_c)
    phicrots[:, 1, 2] = -torch.sin(-dofs.phi_c)
    phicrots[:, 2, 1] = torch.sin(-dofs.phi_c)
    phicrots[:, 2, 2] = torch.cos(-dofs.phi_c)
    theta_axes = torch.matmul(Ms[:, 0:3, 0:3], phicrots)[:, 0:3, 2]

    dsc_ddofs = BondDOF.zeros((nbondatoms,), device=Ms.device)

    # the einsums are doing dot products on stacks of ints
    dsc_ddofs.d[:] = torch.einsum("ij,ij->i", (phi_c_axes, f2s))
    dsc_ddofs.theta[:] = -1 * (
        torch.einsum("ij,ij->i", (theta_axes, f1s))
        + torch.einsum("ij,ij->i", (torch.cross(theta_axes, end_p_pos), f2s))
    )
    dsc_ddofs.phi_p[:] = -1 * (
        torch.einsum("ij,ij->i", (phi_p_axes, f1s))
        + torch.einsum("ij,ij->i", (torch.cross(phi_p_axes, end_p_pos), f2s))
    )
    dsc_ddofs.phi_c[:] = -1 * (
        torch.einsum("ij,ij->i", (phi_c_axes, f1s))
        + torch.einsum("ij,ij->i", (torch.cross(phi_c_axes, end_c_pos), f2s))
    )

    return dsc_ddofs


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

    f1s[:] = f1f2s[:, 0:3]
    f2s[:] = f1f2s[:, 3:6]

    # 3) convert to dscore/dtors
    dsc_ddofs = dofs.clone()
    parentIdx = kintree.parent.to(dtype=torch.long)

    bondSelector = kintree.doftype == NodeType.bond
    dsc_ddofs.bond[bondSelector] = BondDerivatives(
        dofs.bond[bondSelector],
        HTs[bondSelector],
        HTs[parentIdx[bondSelector]],
        f1s[bondSelector],
        f2s[bondSelector],
    )

    jumpSelector = kintree.doftype == NodeType.jump
    dsc_ddofs.jump[jumpSelector] = JumpDerivatives(
        dofs.jump[jumpSelector],
        HTs[jumpSelector],
        HTs[parentIdx[jumpSelector]],
        f1s[jumpSelector],
        f2s[jumpSelector],
    )

    return dsc_ddofs
