from enum import Enum
import attr
import numpy
import numba

import scipy.sparse.csgraph

import torch
from typing import Optional

from tmol.types.functional import validate_args
from tmol.types.torch import Tensor

from .datatypes import NodeType, KinTree, KinDOF, BondDOF, JumpDOF, RefoldData
from .gpu_operations import get_devicendarray, segscan_hts_gpu, segscan_f1f2s_gpu

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


class SegScanStrategy(Enum):
    efficient = "efficient"
    min_depth = "min_depth"
    default = efficient


@validate_args
def SegScanMinDepth(
        data: Tensor(torch.double),
        parents: Tensor(torch.long)[:],
        operator,
):
    """
    Segmented scan code for passing:
        - HT's down the atom tree
        - derivs up the atom tree
    This version implements "Algorithm 1" from
        https://en.wikipedia.org/wiki/Prefix_sum
    This version optimizes depth at the expense of efficiency.
    It will be faster if there are many compute units.
    """
    nelts = data.shape[0]
    N = numpy.ceil(numpy.log2(nelts)
                   )  # this might result in several no-op rounds...

    backPointers = parents
    prevBackPointers = torch.arange(nelts, dtype=torch.long)
    toCalc = (prevBackPointers != backPointers)

    for i in range(int(N)):
        prevBackPointers = backPointers
        operator(data, backPointers, toCalc)
        backPointers = prevBackPointers[prevBackPointers]
        toCalc = (prevBackPointers != backPointers)


@validate_args
def SegScanEfficient(
        data: Tensor(torch.double),
        parents: Tensor(torch.long)[:],
        operator,
        upwards: bool,
):
    """
    Segmented scan code for passing:
        - HT's down the atom tree
        - derivs up the atom tree
    This version implements "Algorithm 2" from
        https://en.wikipedia.org/wiki/Prefix_sum
    This version optimizes efficiency while doubling depth.
    It will be faster if there are few compute units.
    """
    nelts = data.shape[0]

    # calculate depth of each element in tree
    # this logic could be moved into tree construction
    pmat = scipy.sparse.coo_matrix(
        (
            numpy.full(len(parents), 1), (
                parents.numpy(),
                numpy.arange(len(parents)),
            )
        ),
        shape=[len(parents)] * 2,
    )

    treedepth = torch.from_numpy(
        scipy.sparse.csgraph.dijkstra(pmat, indices=[0], unweighted=True)[0]
    ).to(data.device)

    maxdepth = torch.max(treedepth)
    if (upwards):
        treedepth = maxdepth - treedepth

    N = numpy.ceil(numpy.log2(maxdepth + 1))

    # calculation of backpointer array could also
    #   be moved to a precompute step
    backPointers = torch.empty((N + 1, nelts),
                               dtype=torch.long,
                               device=data.device)
    backPointers[0, :] = parents

    # forward pass
    # we need to save backward pointers so we can unroll this function
    for i in range(int(N)):
        if (upwards):
            mask = (treedepth % (2 << i)) == ((1 << i) - 1)
        else:
            mask = (treedepth % (2 << i)) == ((2 << i) - 1)
        operator(data, backPointers[i, :], mask)
        backPointers[i + 1, :] = backPointers[i, :][backPointers[i, :]]

    # backward pass
    for i in range(int(N - 1), -1, -1):
        if (upwards):
            mask = (treedepth % (2 << i)) == ((2 << i) - 1)
        else:
            mask = ((treedepth %
                     (2 << i)) == ((1 << i) - 1)) & (treedepth >= (2 << i))
        operator(data, backPointers[i, :], mask)


@validate_args
def SegScan(
        data: Tensor(torch.double),
        parents: Tensor(torch.long)[:],
        operator,
        upwards: bool,
        scan_strategy: SegScanStrategy = SegScanStrategy.default
):
    if scan_strategy == SegScanStrategy.efficient:
        SegScanEfficient(data, parents, operator, upwards)
    elif scan_strategy == SegScanStrategy.min_depth:
        SegScanMinDepth(data, parents, operator)
    else:
        raise NotImplementedError


@validate_args
def HTcollect(
        HTs: HTArray,
        ptrs: Tensor(torch.long)[:],
        toCalc: Tensor(torch.uint8)[:],
) -> None:
    """segmented scan "down" operator: aggregate HTs"""
    a = HTs[ptrs[toCalc]]
    b = HTs[toCalc]
    res = a @ b
    HTs[toCalc] = res


@validate_args
def Fscollect(
        fs: CoordArray,
        ptrs: Tensor(torch.long)[:],
        toCalc: Tensor(torch.uint8)[:],
) -> None:
    """segmented scan "up" operator: aggregate f1/f2s"""
    offsets = torch.tensor([0, 1, 2], dtype=ptrs.dtype,
                           device=ptrs.device).unsqueeze(0)
    indices = (ptrs.unsqueeze(1) * 3 + offsets)
    fs.put_(indices[toCalc, :], fs[toCalc, :], accumulate=True)


@validate_args
def JumpTransforms(dofs: JumpDOF) -> HTArray:
    """JUMP dofs -> HTs

    jump dofs are _9_ parameters:
     - 3 translational
     - 3 rotational deltas
     - 3 rotational
    Only the rotational deltas should be exposed to minimization

    Translations are represented as an offset in X,Y,Z
    Rotations and rotational deltas are ZYX Euler angles,
        that is, a rotation about Z, then Y, then X
    The HT returned by this function is given by:
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

    Rotational derivatives use the Abe and Go "trick" that allows us to
        easily compute derivatives with respect to rotation about an axis.
    In this case, there are three axes to compute derivatives of:
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
    """
    BOND dofs -> HTs

    each bond has four dofs: [phi_p, theta, d, phi_c]
    in the local frame:
        - phi_p and phi_c are a rotation about x
        - theta is a rotation about z
        - d is a translation along x
    the matrix below is a composition:
        M <- rot(phi_p, [1,0,0]) @ rot(theta, [0,0,1]
           @ trans(d, [1,0,0]) @ rot(phi_c, [1,0,0])
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


@attr.s(frozen=True, auto_attribs=True)
class BackKinResult:
    @classmethod
    @validate_args
    def create(cls, hts: HTArray, dofs: KinDOF):
        return cls(hts, dofs)

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

    return BackKinResult.create(HTs, dofs)


@attr.s(frozen=True, auto_attribs=True)
class ForwardKinResult:
    @classmethod
    @validate_args
    def create(cls, hts: HTArray, coords: CoordArray):
        return cls(hts, coords)

    hts: HTArray
    coords: CoordArray


@validate_args
def forwardKin(
        kintree: KinTree,
        dofs: KinDOF,
        scan_strategy: SegScanStrategy = SegScanStrategy.default
) -> ForwardKinResult:
    """dofs -> HTs, xyzs

      - "forward" kinematics
    """
    natoms = len(dofs)
    assert len(kintree) == len(dofs)

    # 1) local HTs
    HTs = torch.empty([natoms, 4, 4],
                      dtype=HTArray.dtype,
                      device=dofs.raw.device)

    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0
    HTs[0] = torch.eye(4)

    bondSelector = kintree.doftype == NodeType.bond
    HTs[bondSelector] = BondTransforms(dofs.bond[bondSelector])

    jumpSelector = kintree.doftype == NodeType.jump
    HTs[jumpSelector] = JumpTransforms(dofs.jump[jumpSelector])

    # 2) global HTs (rewrite 1->N in-place)
    SegScan(HTs, kintree.parent, HTcollect, False, scan_strategy)

    coords = HTs[:, :3, 3]
    return ForwardKinResult.create(HTs, coords)


@validate_args
def forwardKin2(
        kintree: KinTree,
        refold_data: RefoldData,
        dofs: KinDOF,
        scan_strategy: SegScanStrategy = SegScanStrategy.default
) -> ForwardKinResult:
    """dofs -> HTs, xyzs

      - "forward" kinematics
    """
    natoms = len(dofs)
    assert len(kintree) == len(dofs)

    # 1) local HTs
    HTs = torch.empty([natoms, 4, 4], dtype=torch.double)

    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0
    HTs[0] = torch.eye(4)

    bondSelector = kintree.doftype == NodeType.bond
    HTs[bondSelector] = BondTransforms(dofs.bond[bondSelector])

    jumpSelector = kintree.doftype == NodeType.jump
    HTs[jumpSelector] = JumpTransforms(dofs.jump[jumpSelector])

    if refold_data.natoms != kintree.id.shape[0]:
        # i.e. we are not using the GPU version, and therefore we'll
        # just run an iterative pass over all the HTs using numba's
        # "nopython" compilation mode to make it superfast

        # import time
        # HTscopy = HTs.clone()
        # scan_strategy = SegScanStrategy("efficient")
        # start_time = time.time()
        # for ii in range(10000):
        #     HTscopy[:] = HTs
        #
        #     #SegScan(HTscopy, kintree.parent, HTcollect, False, scan_strategy)
        #     iterative_refold2(HTscopy.numpy(), kintree.parent.numpy())
        # end_time = time.time()
        # print(
        #     "---- refold %f seconds ----" % ((end_time - start_time) / 10000)
        # )

        iterative_refold(HTs.numpy(), kintree.parent.numpy())
    else:
        # 2) numba segmented scan algorithm
        HTs_d = get_devicendarray(HTs)
        segscan_hts_gpu(HTs_d, refold_data)

    coords = HTs[:, :3, 3]
    return ForwardKinResult.create(HTs, coords)


@validate_args
def resolveDerivs(
        kintree: KinTree,
        dofs: KinDOF,
        HTs: HTArray,
        dsc_dx: CoordArray,
        scan_strategy: SegScanStrategy = SegScanStrategy.default
) -> KinDOF:
    """xyz derivs -> dof derivs

    - derivative mapping using Abe and Go approach
    """

    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0

    assert len(kintree) == len(dofs)
    assert len(kintree) == len(HTs)
    assert len(kintree) == len(dsc_dx)

    # 1) local f1/f2s
    Xs = HTs[:, 0:3, 3]

    f1s = torch.cross(Xs, Xs - dsc_dx)
    f2s = dsc_dx.clone()  # clone input buffer before aggregation

    # 2) pass f1/f2s up tree
    SegScan(f1s, kintree.parent, Fscollect, True, scan_strategy)
    SegScan(f2s, kintree.parent, Fscollect, True, scan_strategy)

    #print("frank f1s"); print(f1s)
    #print("frank f2s"); print(f2s)

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


@validate_args
def resolveDerivs2(
        kintree: KinTree, refold_data: RefoldData, dofs: KinDOF, HTs: HTArray,
        dsc_dx: CoordArray
) -> KinDOF:
    """xyz derivs -> dof derivs

    - derivative mapping using Abe and Go approach
    """

    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0

    assert len(kintree) == len(dofs)
    assert len(kintree) == len(HTs)
    assert len(kintree) == len(dsc_dx)

    # 1) local f1/f2s
    Xs = HTs[:, 0:3, 3]

    f1s = torch.cross(Xs, Xs - dsc_dx)
    f2s = dsc_dx.clone()  # clone input buffer before aggregation

    # 2) sum f1/f2s up tree, from leaves toward the root using numba
    # implementation of segmented scan
    f1f2s = torch.cat((f1s, f2s), 1)
    if refold_data.natoms != kintree.id.shape[0]:
        # i.e. we're running on the CPU so we can use an iterative algorithm
        # for accumulating the f1s and f2s

        # import time
        # #f1f2scopy = f1f2s.clone()
        # f1scopy = f1s.clone(); f2scopy = f2s.clone();
        # scan_strategy = SegScanStrategy("efficient")
        # start_time = time.time()
        # for ii in range(1000):
        #     f1scopy[:] = f1s
        #     f2scopy[:] = f2s
        #     SegScan(f1scopy, kintree.parent, Fscollect, True, scan_strategy)
        #     SegScan(f2scopy, kintree.parent, Fscollect, True, scan_strategy)
        #     #iterative_f1f2_summation(f1f2scopy.numpy(), kintree.parent.numpy())
        # end_time = time.time()
        # print(
        #     "---- f1f2 summation %f seconds ----" % ((end_time - start_time) / 1000)
        # )

        iterative_f1f2_summation(f1f2s.numpy(), kintree.parent.numpy())

    else:
        f1f2s_d = get_devicendarray(f1f2s)
        segscan_f1f2s_gpu(f1f2s_d, refold_data)
        #print("f1f2s_gpu"); print(f1f2s_d)
        #print("f1f2s tensor?"); print(f1f2s.numpy())

    f1s[:] = f1f2s[:, 0:3]
    f2s[:] = f1f2s[:, 3:6]

    #print("numba f1s"); print(f1s)
    #print("numba f2s"); print(f2s)

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


@numba.jit(nopython=True)
def iterative_refold(hts, parent):
    for ii in range(1, hts.shape[0]):
        hts[ii, :, :] = hts[parent[ii], :, :] @ hts[ii, :, :]


# @numba.jit(nopython=True)
# def iterative_refold2(hts, parent):
#     for ii in range(1, hts.shape[0]):
#         pii = parent[ii]
#         hts0 = hts[pii, 0, 0] * hts[ii, 0, 0] + hts[pii, 0, 1] * hts[ii, 1, 0] + hts[pii, 0, 2] * hts[ii, 2, 0]
#         hts1 = hts[pii, 0, 0] * hts[ii, 0, 1] + hts[pii, 0, 1] * hts[ii, 1, 1] + hts[pii, 0, 2] * hts[ii, 2, 1]
#         hts2 = hts[pii, 0, 0] * hts[ii, 0, 2] + hts[pii, 0, 1] * hts[ii, 1, 2] + hts[pii, 0, 2] * hts[ii, 2, 2]
#         hts3 = hts[pii, 0, 0] * hts[ii, 0, 3] + hts[pii, 0, 1] * hts[ii, 1, 3] + hts[pii, 0, 2] * hts[ii, 2, 3] + hts[pii, 0, 3]
#
#         hts4 = hts[pii, 1, 0] * hts[ii, 0, 0] + hts[pii, 1, 1] * hts[ii, 1, 0] + hts[pii, 1, 2] * hts[ii, 2, 0]
#         hts5 = hts[pii, 1, 0] * hts[ii, 0, 1] + hts[pii, 1, 1] * hts[ii, 1, 1] + hts[pii, 1, 2] * hts[ii, 2, 1]
#         hts6 = hts[pii, 1, 0] * hts[ii, 0, 2] + hts[pii, 1, 1] * hts[ii, 1, 2] + hts[pii, 1, 2] * hts[ii, 2, 2]
#         hts7 = hts[pii, 1, 0] * hts[ii, 0, 3] + hts[pii, 1, 1] * hts[ii, 1, 3] + hts[pii, 1, 2] * hts[ii, 2, 3] + hts[pii, 1, 3]
#
#         hts8  = hts[pii, 2, 0] * hts[ii, 0, 0] + hts[pii, 2, 1] * hts[ii, 1, 0] + hts[pii, 2, 2] * hts[ii, 2, 0]
#         hts9  = hts[pii, 2, 0] * hts[ii, 0, 1] + hts[pii, 2, 1] * hts[ii, 1, 1] + hts[pii, 2, 2] * hts[ii, 2, 1]
#         hts10 = hts[pii, 2, 0] * hts[ii, 0, 2] + hts[pii, 2, 1] * hts[ii, 1, 2] + hts[pii, 2, 2] * hts[ii, 2, 2]
#         hts11 = hts[pii, 2, 0] * hts[ii, 0, 3] + hts[pii, 2, 1] * hts[ii, 1, 3] + hts[pii, 2, 2] * hts[ii, 2, 3] + hts[pii, 2, 3]
#
#         hts[ii,0,0] = hts0
#         hts[ii,0,1] = hts1
#         hts[ii,0,2] = hts2
#         hts[ii,0,3] = hts3
#
#         hts[ii,1,0] = hts4
#         hts[ii,1,1] = hts5
#         hts[ii,1,2] = hts6
#         hts[ii,1,3] = hts7
#
#         hts[ii,2,0] = hts8
#         hts[ii,2,1] = hts9
#         hts[ii,2,2] = hts10
#         hts[ii,2,3] = hts11


@numba.jit(nopython=True)
def iterative_f1f2_summation(f1f2s, parent):
    for ii in range(f1f2s.shape[0] - 1, 0, -1):
        f1f2s[parent[ii], :] += f1f2s[ii, :]
