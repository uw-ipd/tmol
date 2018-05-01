import enum
import attr
import numpy
from typing import Optional

from tmol.types.functional import validate_args
from tmol.types.array import NDArray

from .datatypes import DOFType, kintree_node_dtype

KinTree = NDArray(kintree_node_dtype)[:]

HTArray = NDArray(float)[:, 4, 4]
VecArray = NDArray(float)[:, 3]

dof_node_dtype = numpy.dtype(
    dict(
        names=["bond",          "jump",        "raw"],
        formats=[(float, 4),    (float, 9),    (float, 9)],
        offsets=[0,             0,             0],
    )
) # yapf: disable


# convenience methods
class BondDOFs(enum.IntEnum):
    phi_p = 0
    theta = enum.auto()
    d = enum.auto()
    phi_c = enum.auto()


class JumpDOFs(enum.IntEnum):
    RBx = 0
    RBy = enum.auto()
    RBz = enum.auto()
    RBdel_alpha = enum.auto()
    RBdel_beta = enum.auto()
    RBdel_gamma = enum.auto()
    RBalpha = enum.auto()
    RBbeta = enum.auto()
    RBgamma = enum.auto()


DOFArray = NDArray(dof_node_dtype)[:]
BondDOFArray = NDArray(float)[:, 4]
JumpDOFArray = NDArray(float)[:, 9]


@validate_args
def HTinv(HTs: HTArray) -> HTArray:
    """helper to quickly invert a HT"""
    N = HTs.shape[0]
    HTinvs = numpy.tile(numpy.identity(4), (N, 1, 1))
    HTinvs[:, :3, :3] = numpy.transpose(HTs[:, :3, :3], (0, 2, 1))
    HTinvs[:, :3, 3] = -numpy.einsum(
        'aij,aj->ai',
        HTinvs[:, :3, :3],
        HTs[:, :3, 3],
    )
    return HTinvs


@validate_args
def SegScan(
        data: numpy.ndarray,
        parents: NDArray(int)[:],
        operator,
        verbose=False,
) -> numpy.ndarray:
    """segmented scan code for passing:

    - HT's down the atom tree
    - derivs up the atom tree
    """
    nelts = data.shape[0]

    # this might result in several extra rounds...
    N = numpy.ceil(numpy.log2(nelts))

    backPointers = parents
    prevBackPointers = numpy.arange(nelts)
    toCalc = (prevBackPointers != backPointers)
    retval = data
    for i in numpy.arange(N):
        prevBackPointers = backPointers
        operator(retval, backPointers, toCalc)
        backPointers = prevBackPointers[prevBackPointers]
        toCalc = (prevBackPointers != backPointers)

    return (retval)


@validate_args
def HTcollect(
        HTs: HTArray,
        ptrs: NDArray(int)[:],
        toCalc: NDArray(bool)[:],
) -> None:
    """segmented scan "down" operator: aggregate HTs"""
    HTs[toCalc] = numpy.matmul(HTs[ptrs][toCalc], HTs[toCalc])


@validate_args
def Fscollect(
        fs: VecArray,
        ptrs: NDArray(int)[:],
        toCalc: NDArray(bool)[:],
) -> None:
    """segmented scan "up" operator: aggregate f1/f2s"""
    numpy.add.at(fs, ptrs[toCalc], fs[toCalc])


@validate_args
def JumpTransforms(dofs: JumpDOFArray) -> HTArray:
    """JUMP dofs -> HTs

    jump dofs are _9_ parameters:
     - 3 translational
     - 3 rotational deltas
     - 3 rotational
    Only the rotational deltas are exposed to minimization
    """
    natoms = dofs.shape[0]

    si = numpy.sin(dofs[:, JumpDOFs.RBdel_alpha])
    sj = numpy.sin(dofs[:, JumpDOFs.RBdel_beta])
    sk = numpy.sin(dofs[:, JumpDOFs.RBdel_gamma])
    ci = numpy.cos(dofs[:, JumpDOFs.RBdel_alpha])
    cj = numpy.cos(dofs[:, JumpDOFs.RBdel_beta])
    ck = numpy.cos(dofs[:, JumpDOFs.RBdel_gamma])
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    Rdelta = numpy.zeros([natoms, 4, 4])
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
    Rdelta[:, 0, 3] = dofs[:, JumpDOFs.RBx]
    Rdelta[:, 1, 3] = dofs[:, JumpDOFs.RBy]
    Rdelta[:, 2, 3] = dofs[:, JumpDOFs.RBz]

    si = numpy.sin(dofs[:, JumpDOFs.RBalpha])
    sj = numpy.sin(dofs[:, JumpDOFs.RBbeta])
    sk = numpy.sin(dofs[:, JumpDOFs.RBgamma])
    ci = numpy.cos(dofs[:, JumpDOFs.RBalpha])
    cj = numpy.cos(dofs[:, JumpDOFs.RBbeta])
    ck = numpy.cos(dofs[:, JumpDOFs.RBgamma])
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    Rglobal = numpy.zeros([natoms, 4, 4])
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

    Ms = numpy.matmul(Rdelta, Rglobal)

    return Ms


@validate_args
def InvJumpTransforms(Ms: HTArray) -> JumpDOFArray:
    """HTs -> JUMP dofs

    this function will always assign rotational delta = 0
    """

    njumpatoms = Ms.shape[0]

    dofs = numpy.empty([njumpatoms, 9])

    dofs[:,
         [JumpDOFs.RBx, JumpDOFs.RBy, JumpDOFs.RBz]] = Ms[:, :3,
                                                          3]  # translation
    dofs[:,
         [JumpDOFs.RBdel_alpha, JumpDOFs.RBdel_beta, JumpDOFs.RBdel_gamma]
         ] = 0  # rotational "delta"

    cys = numpy.sqrt(Ms[:, 0, 0] * Ms[:, 0, 0] + Ms[:, 1, 0] * Ms[:, 1, 0])

    problemSelector = (cys <= 4 * numpy.finfo(float).eps)

    dofs[~problemSelector, JumpDOFs.RBalpha] = numpy.arctan2(
        Ms[~problemSelector, 2, 1], Ms[~problemSelector, 2, 2]
    )
    dofs[~problemSelector, JumpDOFs.RBbeta] = numpy.arctan2(
        -Ms[~problemSelector, 2, 0], cys[~problemSelector]
    )
    dofs[~problemSelector, JumpDOFs.RBgamma] = numpy.arctan2(
        Ms[~problemSelector, 1, 0], Ms[~problemSelector, 0, 0]
    )

    dofs[problemSelector, JumpDOFs.RBalpha] = numpy.arctan2(
        -Ms[problemSelector, 1, 2], Ms[problemSelector, 1, 1]
    )
    dofs[problemSelector, JumpDOFs.RBbeta] = numpy.arctan2(
        -Ms[problemSelector, 2, 0], cys[problemSelector]
    )
    dofs[problemSelector, JumpDOFs.RBgamma] = 0.0

    return dofs


@validate_args
def JumpDerivatives(
        dofs: JumpDOFArray,
        Ms: HTArray,
        Mparents: HTArray,
        f1s: VecArray,
        f2s: VecArray,
) -> JumpDOFArray:
    """compute JUMP derivatives from f1/f2"""
    # trans dofs
    njumpatoms = dofs.shape[0]
    dsc_ddofs = numpy.zeros([njumpatoms, 9])
    x_axes = Mparents[:, 0:3, 0]
    y_axes = Mparents[:, 0:3, 1]
    z_axes = Mparents[:, 0:3, 2]
    dsc_ddofs[:, JumpDOFs.RBx] = numpy.einsum('ij, ij->i', x_axes, f2s)
    dsc_ddofs[:, JumpDOFs.RBy] = numpy.einsum('ij, ij->i', y_axes, f2s)
    dsc_ddofs[:, JumpDOFs.RBz] = numpy.einsum('ij, ij->i', z_axes, f2s)

    end_pos = Ms[:, 0:3, 3]
    rotdof3_axes = -Mparents[:, 0:3, 2]

    zrots = numpy.zeros([njumpatoms, 3, 3])
    zrots[:, 0, 0] = numpy.cos(dofs[:, 5])
    zrots[:, 0, 1] = -numpy.sin(dofs[:, 5])
    zrots[:, 1, 0] = numpy.sin(dofs[:, 5])
    zrots[:, 1, 1] = numpy.cos(dofs[:, 5])
    zrots[:, 2, 2] = 1
    rotdof2_axes = -numpy.matmul(Mparents[:, 0:3, 0:3], zrots)[:, 0:3, 1]

    yrots = numpy.empty([njumpatoms, 3, 3])
    yrots[:, 0, 0] = numpy.cos(-dofs[:, 4])
    yrots[:, 0, 2] = -numpy.sin(-dofs[:, 4])
    yrots[:, 1, 1] = 1
    yrots[:, 2, 0] = numpy.sin(-dofs[:, 4])
    yrots[:, 2, 2] = numpy.cos(-dofs[:, 4])
    rotdof1_axes = -numpy.matmul(
        numpy.matmul(Mparents[:, 0:3, 0:3], zrots), yrots
    )[:, 0:3, 0]

    dsc_ddofs[:, JumpDOFs.RBdel_alpha] = (
        numpy.einsum('ij, ij->i', rotdof1_axes, f1s) +
        numpy.einsum('ij, ij->i', numpy.cross(rotdof1_axes, end_pos), f2s)
    )
    dsc_ddofs[:, JumpDOFs.RBdel_beta] = (
        numpy.einsum('ij, ij->i', rotdof2_axes, f1s) +
        numpy.einsum('ij, ij->i', numpy.cross(rotdof2_axes, end_pos), f2s)
    )
    dsc_ddofs[:, JumpDOFs.RBdel_gamma] = (
        numpy.einsum('ij, ij->i', rotdof3_axes, f1s) +
        numpy.einsum('ij, ij->i', numpy.cross(rotdof3_axes, end_pos), f2s)
    )

    return dsc_ddofs


@validate_args
def BondTransforms(dofs: BondDOFArray) -> HTArray:
    """BOND dofs -> HTs"""
    natoms = dofs.shape[0]

    cpp = numpy.cos(dofs[:, BondDOFs.phi_p])
    spp = numpy.sin(dofs[:, BondDOFs.phi_p])
    cpc = numpy.cos(dofs[:, BondDOFs.phi_c])
    spc = numpy.sin(dofs[:, BondDOFs.phi_c])
    cth = numpy.cos(dofs[:, BondDOFs.theta])
    sth = numpy.sin(dofs[:, BondDOFs.theta])
    d = dofs[:, BondDOFs.d]

    # rot(ph_p, +x) * rot(th, +z) * trans(d, +x) * rot(ph_c, +x)
    Ms = numpy.zeros([natoms, 4, 4])
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
    Ms[:, 3, 3] = 1

    return Ms


@validate_args
def InvBondTransforms(Ms: HTArray) -> BondDOFArray:
    """HTs -> BOND dofs"""
    nbondatoms = Ms.shape[0]

    dofs = numpy.empty([nbondatoms, 4])

    # d is always the same logic
    dofs[:, BondDOFs.d] = numpy.sqrt(numpy.square(Ms[:, :3, 3]).sum(axis=1))

    # when theta == 0, phip and phic are about same axis
    # we (arbitrarily) put all the movement into phic
    theta0_selector = (
        numpy.abs(Ms[:, 0, 0] - 1) <= 4 * numpy.finfo(float).eps
    )
    dofs[theta0_selector, BondDOFs.phi_p] = 0.0
    dofs[theta0_selector, BondDOFs.phi_c] = numpy.arctan2(
        Ms[theta0_selector, 2, 1], Ms[theta0_selector, 1, 1]
    )
    dofs[theta0_selector, BondDOFs.theta] = 0

    # otherwise, use the general case
    dofs[~theta0_selector, BondDOFs.phi_p] = numpy.arctan2(
        Ms[~theta0_selector, 2, 0], Ms[~theta0_selector, 1, 0]
    )
    dofs[~theta0_selector, BondDOFs.phi_c] = numpy.arctan2(
        Ms[~theta0_selector, 0, 2], -Ms[~theta0_selector, 0, 1]
    )
    dofs[~theta0_selector, BondDOFs.theta] = numpy.arctan2(
        numpy.sqrt(
            Ms[~theta0_selector, 0, 1] * Ms[~theta0_selector, 0, 1] +
            Ms[~theta0_selector, 0, 2] * Ms[~theta0_selector, 0, 2]
        ), Ms[~theta0_selector, 0, 0]
    )

    return dofs


@validate_args
def BondDerivatives(
        dofs: BondDOFArray,
        Ms: HTArray,
        Mparents: HTArray,
        f1s: VecArray,
        f2s: VecArray,
) -> BondDOFArray:
    """compute BOND derivatives from f1/f2"""
    nbondatoms = dofs.shape[0]

    end_p_pos = Mparents[:, 0:3, 3]
    phi_p_axes = Mparents[:, 0:3, 0]
    theta_axes = Ms[:, 0:3, 2]
    end_c_pos = Ms[:, 0:3, 3]
    phi_c_axes = Ms[:, 0:3, 0]

    dsc_ddofs = numpy.zeros([nbondatoms, 4])

    dsc_ddofs[:, BondDOFs.d] = numpy.einsum('ij, ij->i', phi_c_axes, f2s)
    dsc_ddofs[:, BondDOFs.theta] = -1 * (
        numpy.einsum('ij, ij->i', theta_axes, f1s) +
        numpy.einsum('ij, ij->i', numpy.cross(theta_axes, end_p_pos), f2s)
    )
    dsc_ddofs[:, BondDOFs.phi_p] = -1 * (
        numpy.einsum('ij, ij->i', phi_p_axes, f1s) +
        numpy.einsum('ij, ij->i', numpy.cross(phi_p_axes, end_p_pos), f2s)
    )
    dsc_ddofs[:, BondDOFs.phi_c] = -1 * (
        numpy.einsum('ij, ij->i', phi_c_axes, f1s) +
        numpy.einsum('ij, ij->i', numpy.cross(phi_c_axes, end_c_pos), f2s)
    )

    return dsc_ddofs


@validate_args
def HTs_from_frames(
        Cs: VecArray,
        Xs: VecArray,
        Ys: VecArray,
        Zs: VecArray,
        out: Optional[HTArray] = None,
) -> HTArray:
    """xyzs -> HTs"""
    natoms = Cs.shape[0]

    def unit_norm(v):
        return v / numpy.linalg.norm(v, axis=-1, keepdims=True)

    if out is None:
        out = numpy.zeros([natoms, 4, 4])
    else:
        assert out.shape[0] == natoms

    xaxis = out[:, :3, 0]
    yaxis = out[:, :3, 1]
    zaxis = out[:, :3, 2]
    center = out[:, :3, 3]

    xaxis[:] = unit_norm(Xs - Ys)
    zaxis[:] = unit_norm(numpy.cross(xaxis, Zs - Xs))
    yaxis[:] = unit_norm(numpy.cross(zaxis, xaxis))
    center[:] = Cs

    out[:, 3] = [0, 0, 0, 1]

    return (out)


@attr.s(frozen=True, auto_attribs=True)
class BackKinResult:
    @classmethod
    @validate_args
    def create(cls, hts: HTArray, dofs: DOFArray):
        return cls(hts, dofs)

    hts: HTArray
    dofs: DOFArray


@validate_args
def backwardKin(kintree: KinTree, coords: VecArray) -> BackKinResult:
    """xyzs -> HTs, dofs

      - "backward" kinematics
    """
    natoms = coords.shape[0]

    parents = kintree["parent"]

    # 1) global HTs
    HTs = numpy.empty((natoms, 4, 4))

    assert kintree[0]["doftype"] == DOFType.root
    assert kintree["parent"][0] == 0
    assert numpy.all(coords[0] == 0) or numpy.all(numpy.isnan(coords[0]))

    HTs = numpy.empty((natoms, 4, 4))
    HTs[0] = numpy.identity(4)
    HTs_from_frames(
        coords[1:],
        coords[kintree[1:]["frame_x"], :],
        coords[kintree[1:]["frame_y"], :],
        coords[kintree[1:]["frame_z"], :],
        out=HTs[1:],
    )

    # 2) local HTs
    localHTs = numpy.empty([natoms, 4, 4])
    localHTs[1:] = numpy.matmul(HTinv(HTs[parents[1:], :, :]), HTs[1:, :, :])

    # 3) dofs
    dofs = numpy.zeros([natoms], dof_node_dtype)

    bondSelector = (kintree["doftype"] == DOFType.bond)
    bondSelector[0] = False
    dofs["bond"][bondSelector] = InvBondTransforms(localHTs[bondSelector])

    jumpSelector = (kintree["doftype"] == DOFType.jump)
    jumpSelector[0] = False
    dofs["jump"][jumpSelector] = InvJumpTransforms(localHTs[jumpSelector])

    return BackKinResult.create(HTs, dofs)


@attr.s(frozen=True, auto_attribs=True)
class ForwardKinResult:
    @classmethod
    @validate_args
    def create(cls, hts: HTArray, coords: VecArray):
        return cls(hts, coords)

    hts: HTArray
    coords: VecArray


@validate_args
def forwardKin(kintree: KinTree, dofs: DOFArray) -> ForwardKinResult:
    """dofs -> HTs, xyzs

      - "forward" kinematics
    """
    natoms = dofs.shape[0]

    parents = kintree["parent"]

    # 1) local HTs
    HTs = numpy.empty([natoms, 4, 4])

    assert kintree[0]["doftype"] == DOFType.root
    assert kintree["parent"][0] == 0
    HTs[0] = numpy.identity(4)

    bondSelector = (kintree["doftype"] == DOFType.bond)
    HTs[bondSelector] = BondTransforms(dofs["bond"][bondSelector])

    jumpSelector = (kintree["doftype"] == DOFType.jump)
    HTs[jumpSelector] = JumpTransforms(dofs["jump"][jumpSelector])

    # 2) global HTs (rewrite 1->N in-place)
    SegScan(HTs, parents, HTcollect)

    coords = numpy.zeros([natoms, 3])
    coords = numpy.matmul(HTs, [0, 0, 0, 1])[:, :3]
    return ForwardKinResult.create(HTs, coords)


@validate_args
def resolveDerivs(
        kintree: KinTree,
        dofs: DOFArray,
        HTs: HTArray,
        dsc_dx: VecArray,
) -> DOFArray:
    """xyz derivs -> dof derivs

    - derivative mapping using Abe and Go approach
    """

    parents = kintree["parent"]

    assert kintree[0]["doftype"] == DOFType.root
    assert kintree["parent"][0] == 0

    # 1) local f1/f2s
    Xs = HTs[:, 0:3, 3]
    f1s = numpy.cross(Xs, Xs - dsc_dx)
    f2s = dsc_dx

    # 2) pass f1/f2s up tree
    SegScan(f1s, parents, Fscollect)
    SegScan(f2s, parents, Fscollect)

    # 3) convert to dscore/dtors
    dsc_ddofs = numpy.zeros_like(dofs)

    bondSelector = (kintree["doftype"] == DOFType.bond)
    dsc_ddofs["bond"][bondSelector] = BondDerivatives(
        dofs["bond"][bondSelector],
        HTs[bondSelector],
        HTs[parents[bondSelector]],
        f1s[bondSelector],
        f2s[bondSelector],
    )

    jumpSelector = (kintree["doftype"] == DOFType.jump)
    dsc_ddofs["jump"][jumpSelector] = JumpDerivatives(
        dofs["jump"][jumpSelector],
        HTs[jumpSelector],
        HTs[parents[jumpSelector]],
        f1s[jumpSelector],
        f2s[jumpSelector],
    )

    return dsc_ddofs
