import warnings
import attr
import numpy

import torch
from typing import Optional

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor
from tmol.types.attrs import ValidateAttrs

from .datatypes import NodeType, KinTree, KinDOF, BondDOF, JumpDOF
from .gpu_operations import GPUKinTreeReordering
from .cpu_operations import iterative_refold, iterative_f1f2_summation

HTArray = Tensor(torch.double)[:, 4, 4]
CoordArray = Tensor(torch.double)[:, 3]
EPS = 1e-6


@validate_args
def HTinv(HTs: HTArray) -> HTArray:
    """helper to quickly invert a HT stack"""
    N = HTs.shape[0]
    HTinvs = torch.eye(4).expand(N, -1, -1)

    #fd: not sure why but the line below does not work...
    #HTinvs[:, :3, :3] = torch.transpose(HTs[:, :3, :3], 2, 1)

    # einsum is matrix/vector mult
    HTinvs = torch.transpose(HTs, 2, 1)
    HTinvs[:, :3, 3] = -torch.einsum(
        'aij,aj->ai',
        (HTinvs[:, :3, :3], HTs[:, :3, 3]),
    )
    HTinvs[:, 3, :3] = 0
    return HTinvs


@validate_args
def JumpTransforms(dofs: JumpDOF) -> HTArray:
    """JUMP dofs -> HTs

    jump dofs are _9_ parameters::

     - 3 translational
     - 3 rotational deltas
     - 3 rotational

    Only the rotational deltas should be exposed to minimization

    Translations are represented as an offset in X,Y,Z

    Rotations and rotational deltas are ZYX Euler angles. That is, a rotation
    about Z, then Y, then X.

    The HT returned by this function is given by::

        M = trans( RBx, RBy, RBz)
            @ roteuler( RBdel_alpha, RBdel_alpha, RBdel_alpha)
            @ roteuler( RBalpha, RBalpha, RBalpha)

    RBdel_* is meant to be reset to zero at the beginning of a minimization
        trajectory, as when parameters are near 0, the rotational space
        is well-behaved.
    """
    natoms, = dofs.shape

    si = torch.sin(dofs.RBdel_alpha)
    sj = torch.sin(dofs.RBdel_beta)
    sk = torch.sin(dofs.RBdel_gamma)
    ci = torch.cos(dofs.RBdel_alpha)
    cj = torch.cos(dofs.RBdel_beta)
    ck = torch.cos(dofs.RBdel_gamma)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    Rdelta = torch.zeros((natoms, 4, 4),
                         dtype=HTArray.dtype,
                         device=dofs.raw.device)
    Rdelta[:, 0, 0] = cj * ck
    Rdelta[:, 0, 1] = sj * sc - cs
    Rdelta[:, 0, 2] = sj * cc + ss
    Rdelta[:, 1, 0] = cj * sk
    Rdelta[:, 1, 1] = sj * ss + cc
    Rdelta[:, 1, 2] = sj * cs - sc
    Rdelta[:, 2, 0] = -sj
    Rdelta[:, 2, 1] = cj * si
    Rdelta[:, 2, 2] = cj * ci
    Rdelta[:, 3, 3] = 1

    # translational dofs
    Rdelta[:, 0, 3] = dofs.RBx
    Rdelta[:, 1, 3] = dofs.RBy
    Rdelta[:, 2, 3] = dofs.RBz

    si = torch.sin(dofs.RBalpha)
    sj = torch.sin(dofs.RBbeta)
    sk = torch.sin(dofs.RBgamma)
    ci = torch.cos(dofs.RBalpha)
    cj = torch.cos(dofs.RBbeta)
    ck = torch.cos(dofs.RBgamma)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    Rglobal = torch.zeros((natoms, 4, 4),
                          dtype=HTArray.dtype,
                          device=dofs.raw.device)
    Rglobal[:, 0, 0] = cj * ck
    Rglobal[:, 0, 1] = sj * sc - cs
    Rglobal[:, 0, 2] = sj * cc + ss
    Rglobal[:, 1, 0] = cj * sk
    Rglobal[:, 1, 1] = sj * ss + cc
    Rglobal[:, 1, 2] = sj * cs - sc
    Rglobal[:, 2, 0] = -sj
    Rglobal[:, 2, 1] = cj * si
    Rglobal[:, 2, 2] = cj * ci
    Rglobal[:, 3, 3] = 1

    Ms = torch.matmul(Rdelta, Rglobal)

    return Ms


@validate_args
def InvJumpTransforms(Ms: HTArray) -> JumpDOF:
    """HTs -> JUMP dofs

    Given the matrix definition in JumpTransforms, we calculate the dofs
    that give rise to this HT.

    A special case handles the problematic region where cos(beta)=0.
        In this case, the alpha and gamma rotation are coincident so
        we assign all rotation to alpha.

    Since RB and RBdel are redundant, this function always returns its
        non-zero components into RB, and RBdel is always 0
    """

    njumpatoms = Ms.shape[0]

    dofs = JumpDOF.empty(njumpatoms, device=Ms.device)

    dofs.RBx[:] = Ms[:, 0, 3]
    dofs.RBy[:] = Ms[:, 1, 3]
    dofs.RBz[:] = Ms[:, 2, 3]

    dofs.RBdel_alpha[:] = 0
    dofs.RBdel_beta[:] = 0
    dofs.RBdel_gamma[:] = 0

    cys = torch.sqrt(Ms[:, 0, 0] * Ms[:, 0, 0] + Ms[:, 1, 0] * Ms[:, 1, 0])

    problemSelector = (cys <= EPS)

    dofs.RBalpha[~problemSelector] = torch.atan2(
        Ms[~problemSelector, 2, 1], Ms[~problemSelector, 2, 2]
    )
    dofs.RBbeta[~problemSelector] = torch.atan2(
        -Ms[~problemSelector, 2, 0], cys[~problemSelector]
    )
    dofs.RBgamma[~problemSelector] = torch.atan2(
        Ms[~problemSelector, 1, 0], Ms[~problemSelector, 0, 0]
    )

    dofs.RBalpha[problemSelector] = torch.atan2(
        -Ms[problemSelector, 1, 2], Ms[problemSelector, 1, 1]
    )
    dofs.RBbeta[problemSelector] = torch.atan2(
        -Ms[problemSelector, 2, 0], cys[problemSelector]
    )
    dofs.RBgamma[problemSelector] = 0.0

    return dofs


@validate_args
def JumpDerivatives(
        dofs: JumpDOF,
        Ms: HTArray,
        Mparents: HTArray,
        f1s: CoordArray,
        f2s: CoordArray,
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
    dsc_ddofs = JumpDOF.zeros((njumpatoms, ), device=dofs.raw.device)

    x_axes = Mparents[:, 0:3, 0]
    y_axes = Mparents[:, 0:3, 1]
    z_axes = Mparents[:, 0:3, 2]

    # einsums here are taking dot products of the vector stacks
    dsc_ddofs.RBx[:] = torch.einsum('ij,ij->i', (x_axes, f2s))
    dsc_ddofs.RBy[:] = torch.einsum('ij,ij->i', (y_axes, f2s))
    dsc_ddofs.RBz[:] = torch.einsum('ij,ij->i', (z_axes, f2s))

    end_pos = Ms[:, 0:3, 3]
    rotdof3_axes = -Mparents[:, 0:3, 2]

    zrots = torch.zeros([njumpatoms, 3, 3],
                        dtype=torch.double,
                        device=dofs.raw.device)
    zrots[:, 0, 0] = torch.cos(dofs.RBdel_gamma)
    zrots[:, 0, 1] = -torch.sin(dofs.RBdel_gamma)
    zrots[:, 1, 0] = torch.sin(dofs.RBdel_gamma)
    zrots[:, 1, 1] = torch.cos(dofs.RBdel_gamma)
    zrots[:, 2, 2] = 1
    rotdof2_axes = -torch.matmul(Mparents[:, 0:3, 0:3], zrots)[:, 0:3, 1]

    yrots = torch.zeros([njumpatoms, 3, 3],
                        dtype=torch.double,
                        device=dofs.raw.device)
    yrots[:, 0, 0] = torch.cos(-dofs.RBdel_beta)
    yrots[:, 0, 2] = -torch.sin(-dofs.RBdel_beta)
    yrots[:, 1, 1] = 1
    yrots[:, 2, 0] = torch.sin(-dofs.RBdel_beta)
    yrots[:, 2, 2] = torch.cos(-dofs.RBdel_beta)
    rotdof1_axes = -torch.matmul(
        torch.matmul(Mparents[:, 0:3, 0:3], zrots), yrots
    )[:, 0:3, 0]

    # einsums here are taking dot products of the vector stacks
    dsc_ddofs.RBdel_alpha[:] = (
        torch.einsum('ij,ij->i',
                     (rotdof1_axes, f1s)) +
        torch.einsum('ij,ij->i',
                     (torch.cross(rotdof1_axes, end_pos), f2s))
    )
    dsc_ddofs.RBdel_beta[:] = (
        torch.einsum('ij,ij->i',
                     (rotdof2_axes, f1s)) +
        torch.einsum('ij,ij->i',
                     (torch.cross(rotdof2_axes, end_pos), f2s))
    )
    dsc_ddofs.RBdel_gamma[:] = (
        torch.einsum('ij,ij->i',
                     (rotdof3_axes, f1s)) +
        torch.einsum('ij,ij->i',
                     (torch.cross(rotdof3_axes, end_pos), f2s))
    )

    return dsc_ddofs


@validate_args
def BondTransforms(dofs: BondDOF) -> HTArray:
    """Bond HTs from bond dofs.

    each bond has four dofs: [phi_p, theta, d, phi_c]

    in the local frame:
        - phi_p and phi_c are a rotation about x
        - theta is a rotation about z
        - d is a translation along x

    the matrix below is a composition::

        M = (
            rot(phi_p, [1,0,0])
            @ rot(theta, [0,0,1]
            @ trans(d, [1,0,0])
            @ rot(phi_c, [1,0,0])
        )
    """
    natoms, = dofs.shape

    cpp = torch.cos(dofs.phi_p)
    spp = torch.sin(dofs.phi_p)
    cpc = torch.cos(dofs.phi_c)
    spc = torch.sin(dofs.phi_c)
    cth = torch.cos(dofs.theta)
    sth = torch.sin(dofs.theta)
    d = dofs.d

    # rot(ph_p, +x) * rot(th, +z) * trans(d, +x) * rot(ph_c, +x)
    Ms = torch.empty([natoms, 4, 4],
                     dtype=HTArray.dtype,
                     device=dofs.raw.device)
    Ms[:, 0, 0] = cth
    Ms[:, 0, 1] = -cpc * sth
    Ms[:, 0, 2] = spc * sth
    Ms[:, 0, 3] = d * cth
    Ms[:, 1, 0] = cpp * sth
    Ms[:, 1, 1] = cpc * cpp * cth - spc * spp
    Ms[:, 1, 2] = -cpp * cth * spc - cpc * spp
    Ms[:, 1, 3] = d * cpp * sth
    Ms[:, 2, 0] = spp * sth
    Ms[:, 2, 1] = cpp * spc + cpc * cth * spp
    Ms[:, 2, 2] = cpc * cpp - cth * spc * spp
    Ms[:, 2, 3] = d * spp * sth
    Ms[:, 3, 0] = 0
    Ms[:, 3, 1] = 0
    Ms[:, 3, 2] = 0
    Ms[:, 3, 3] = 1

    return Ms


@validate_args
def InvBondTransforms(Ms: HTArray) -> BondDOF:
    """
    HTs -> BOND dofs

    Given the matrix definition in BondTransforms, we calculate the dofs
    that give rise to this HT.

    A special case below handles a "singularity," that is, a configuration
    where there are multiple parameterizations that give the same HT

    Specifically, when theta==0, the rx rotation can be put into
    phi_c or phi_p (we use phi_c)
    """
    nbondatoms = Ms.shape[0]

    dofs = BondDOF.empty(nbondatoms, device=Ms.device)

    # d is always the same logic
    dofs.d[:] = Ms[:, :3, 3].norm(dim=1)

    # when theta == 0, phip and phic are about same axis
    # we (arbitrarily) put all the movement into phic
    theta0_selector = (torch.abs(Ms[:, 0, 0] - 1) <= EPS)
    dofs.phi_p[theta0_selector] = 0.0
    dofs.phi_c[theta0_selector] = torch.atan2(
        Ms[theta0_selector, 2, 1], Ms[theta0_selector, 1, 1]
    )
    dofs.theta[theta0_selector] = 0

    # otherwise, use the general case
    dofs.phi_p[~theta0_selector] = torch.atan2(
        Ms[~theta0_selector, 2, 0], Ms[~theta0_selector, 1, 0]
    )
    dofs.phi_c[~theta0_selector] = torch.atan2(
        Ms[~theta0_selector, 0, 2], -Ms[~theta0_selector, 0, 1]
    )

    dofs.theta[~theta0_selector] = torch.atan2(
        torch.sqrt(
            Ms[~theta0_selector, 0, 1] * Ms[~theta0_selector, 0, 1] +
            Ms[~theta0_selector, 0, 2] * Ms[~theta0_selector, 0, 2]
        ), Ms[~theta0_selector, 0, 0]
    )

    return dofs


@validate_args
def BondDerivatives(
        dofs: BondDOF,
        Ms: HTArray,
        Mparents: HTArray,
        f1s: CoordArray,
        f2s: CoordArray,
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

    dsc_ddofs = BondDOF.zeros((nbondatoms, ), device=Ms.device)

    # the einsums are doing dot products on stacks of ints
    dsc_ddofs.d[:] = torch.einsum('ij,ij->i', (phi_c_axes, f2s))
    dsc_ddofs.theta[:] = -1 * (
        torch.einsum('ij,ij->i',
                     (theta_axes, f1s)) +
        torch.einsum('ij,ij->i',
                     (torch.cross(theta_axes, end_p_pos), f2s))
    )
    dsc_ddofs.phi_p[:] = -1 * (
        torch.einsum('ij,ij->i',
                     (phi_p_axes, f1s)) +
        torch.einsum('ij,ij->i',
                     (torch.cross(phi_p_axes, end_p_pos), f2s))
    )
    dsc_ddofs.phi_c[:] = -1 * (
        torch.einsum('ij,ij->i',
                     (phi_c_axes, f1s)) +
        torch.einsum('ij,ij->i',
                     (torch.cross(phi_c_axes, end_c_pos), f2s))
    )

    return dsc_ddofs


def DOFTransforms(dof_types: Tensor(torch.int)[:], dofs: KinDOF) -> HTArray:
    """Calculate HT representation of given dofs."""
    HTs = torch.empty([len(dofs), 4, 4],
                      dtype=HTArray.dtype,
                      device=dofs.raw.device)

    rootSelector = dof_types == NodeType.root
    HTs[rootSelector] = torch.eye(4, dtype=HTs.dtype, device=HTs.device)

    bondSelector = dof_types == NodeType.bond
    HTs[bondSelector] = BondTransforms(dofs.bond[bondSelector])

    jumpSelector = dof_types == NodeType.jump
    HTs[jumpSelector] = JumpTransforms(dofs.jump[jumpSelector])

    return HTs


@validate_args
def HTs_from_frames(
        Cs: CoordArray,
        Xs: CoordArray,
        Ys: CoordArray,
        Zs: CoordArray,
        out: Optional[HTArray] = None,
) -> HTArray:
    """xyzs -> HTs"""
    natoms = Cs.shape[0]

    def unit_norm(v):
        return v / torch.norm(v, dim=-1, keepdim=True)

    if out is None:
        out = torch.zeros([natoms, 4, 4], dtype=torch.double, device=Cs.device)
    else:
        assert out.shape[0] == natoms

    xaxis = out[:, :3, 0]
    yaxis = out[:, :3, 1]
    zaxis = out[:, :3, 2]
    center = out[:, :3, 3]

    xaxis[:] = unit_norm(Xs - Ys)
    zaxis[:] = unit_norm(torch.cross(xaxis, Zs - Xs))
    yaxis[:] = unit_norm(torch.cross(zaxis, xaxis))
    center[:] = Cs

    out[:, 3] = torch.tensor([0, 0, 0, 1], device=out.device)

    return (out)


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

    # fd: not sure of a torch isnan check?
    #assert (torch.norm(coords[0, :], dim=-1) == 0)

    HTs = torch.empty((natoms, 4, 4),
                      dtype=HTArray.dtype,
                      device=coords.device)
    HTs[0] = torch.eye(4)
    HTs_from_frames(
        coords[1:],
        coords[kintree.frame_x[1:], :],
        coords[kintree.frame_y[1:], :],
        coords[kintree.frame_z[1:], :],
        out=HTs[1:],
    )

    # 2) local HTs
    localHTs = torch.empty((natoms, 4, 4),
                           dtype=HTArray.dtype,
                           device=coords.device)
    localHTs[0] = torch.eye(4)
    localHTs[1:] = torch.matmul(
        HTinv(HTs[kintree.parent[1:].squeeze(), :, :]), HTs[1:, :, :]
    )

    # 3) dofs
    dofs = KinDOF.full(natoms, numpy.nan, device=coords.device)

    bondSelector = kintree.doftype == NodeType.bond
    dofs.bond[bondSelector] = InvBondTransforms(localHTs[bondSelector])

    jumpSelector = kintree.doftype == NodeType.jump
    dofs.jump[jumpSelector] = InvJumpTransforms(localHTs[jumpSelector])

    return BackKinResult(HTs, dofs)


@attr.s(frozen=True, auto_attribs=True, slots=True)
class ForwardKinResult:
    hts: HTArray
    coords: CoordArray


@validate_args
def forwardKin(
        kintree: KinTree,
        dofs: KinDOF,
) -> ForwardKinResult:
    """dofs -> HTs, xyzs

      - "forward" kinematics
    """
    assert len(kintree) == len(dofs)
    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0

    # 1) Calculate local inter-node HTs from dofs
    HTs = DOFTransforms(kintree.doftype, dofs)

    # 2) Accumulate local HTs into global HTs (rewrite 1->N in-place)
    if HTs.device.type == "cuda":
        (
            GPUKinTreeReordering.for_kintree(kintree)
            .refold_ordering.segscan_hts(HTs, inplace=True)
        )
    else:
        iterative_refold(HTs, kintree.parent, inplace=True)

    coords = HTs[:, :3, 3]
    return ForwardKinResult(HTs, coords)


@validate_args
def resolveDerivs(
        kintree: KinTree,
        dofs: KinDOF,
        HTs: HTArray,
        dsc_dx: CoordArray,
) -> KinDOF:
    """xyz derivs -> dof derivs

    - derivative mapping using Abe and Go approach
    """

    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0

    assert len(kintree) == len(dofs)
    assert len(kintree) == len(HTs)
    assert len(kintree) == len(dsc_dx)

    if dsc_dx.requires_grad:
        # Can not render higher-order derivatives for kinematic operations,
        # as is generated with backward(create_graph=True) is called.
        # Ignore/nan rather than erroring to support torch gradcheck.
        warnings.warn(
            "dsc_dx.requires_grad in resolveDerivs not supported, "
            "will return nan gradient"
        )
        dsc_dx.register_hook(lambda grad: grad.new_full(numpy.nan))
        dsc_dx = dsc_dx.detach()

    # 1) local f1/f2s
    Xs = HTs[:, 0:3, 3]

    f1s = torch.cross(Xs, Xs - dsc_dx)
    f2s = dsc_dx.clone()  # clone input buffer before aggregation

    # 2) pass f1/f2s up tree
    f1f2s = torch.cat((f1s, f2s), 1)

    if f1f2s.device.type == "cuda":
        (
            GPUKinTreeReordering.for_kintree(kintree)
            .derivsum_ordering.segscan_f1f2s(f1f2s, inplace=True)
        )

    else:
        iterative_f1f2_summation(f1f2s, kintree.parent, inplace=True)

    f1s[:] = f1f2s[:, 0:3]
    f2s[:] = f1f2s[:, 3:6]

    # 3) convert to dscore/dtors
    dsc_ddofs = dofs.clone()

    bondSelector = kintree.doftype == NodeType.bond
    dsc_ddofs.bond[bondSelector] = BondDerivatives(
        dofs.bond[bondSelector],
        HTs[bondSelector],
        HTs[kintree.parent[bondSelector]],
        f1s[bondSelector],
        f2s[bondSelector],
    )

    jumpSelector = kintree.doftype == NodeType.jump
    dsc_ddofs.jump[jumpSelector] = JumpDerivatives(
        dofs.jump[jumpSelector],
        HTs[jumpSelector],
        HTs[kintree.parent[jumpSelector]],
        f1s[jumpSelector],
        f2s[jumpSelector],
    )

    return dsc_ddofs
