import numpy

# doftypes
BOND = 1
JUMP = 2

# data structure describing the atom-level kinematics of a molecular system
kintree_node_dtype = numpy.dtype([
    ("atom_name", numpy.str, 4),
    ("resnum", numpy.int),
    ("doftype", numpy.int),
    ("parent", numpy.int),
    ("frame", numpy.int, 3),
])


def HTinv(HTs):
    """helper to quickly invert a HT"""
    N = HTs.shape[0]
    HTinvs = numpy.tile(numpy.identity(4), (N, 1, 1))
    HTinvs[:, :3, :3] = numpy.transpose(HTs[:, :3, :3], (0, 2, 1))
    HTinvs[:, :3, 3
           ] = -numpy.einsum('aij,aj->ai', HTinvs[:, :3, :3], HTs[:, :3, 3])
    return HTinvs


def SegScan(data, parents, operator, verbose=False):
    """segmented scan code for passing:

    - HT's down the atom tree
    - derivs up the atom tree
    """
    nelts = data.shape[0]
    N = numpy.ceil(numpy.log2(nelts)
                   )  # this might result in several extra rounds...

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


def HTcollect(HTs, ptrs, toCalc):
    """segmented scan "down" operator: aggregate HTs"""
    HTs[toCalc] = numpy.matmul(HTs[ptrs][toCalc], HTs[toCalc])


def Fscollect(fs, ptrs, toCalc):
    """segmented scan "up" operator: aggregate f1/f2s"""
    numpy.add.at(fs, ptrs[toCalc], fs[toCalc])


def JumpTransforms(dofs):
    """JUMP dofs -> HTs

    jump dofs are _9_ parameters:
     - 3 translational
     - 3 rotational deltas
     - 3 rotational
    Only the rotational deltas are exposed to minimization
    """
    natoms = dofs.shape[0]

    si = numpy.sin(dofs[:, 3])
    sj = numpy.sin(dofs[:, 4])
    sk = numpy.sin(dofs[:, 5])
    ci = numpy.cos(dofs[:, 3])
    cj = numpy.cos(dofs[:, 4])
    ck = numpy.cos(dofs[:, 5])
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
    Rdelta[:, 0, 3] = dofs[:, 0]
    Rdelta[:, 1, 3] = dofs[:, 1]
    Rdelta[:, 2, 3] = dofs[:, 2]
    Rdelta[:, 3, 3] = 1

    si = numpy.sin(dofs[:, 6])
    sj = numpy.sin(dofs[:, 7])
    sk = numpy.sin(dofs[:, 8])
    ci = numpy.cos(dofs[:, 6])
    cj = numpy.cos(dofs[:, 7])
    ck = numpy.cos(dofs[:, 8])
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


def InvJumpTransforms(Ms):
    """HTs -> JUMP dofs

    this function will always assign rotational delta = 0
    """

    njumpatoms = Ms.shape[0]

    dofs = numpy.empty([njumpatoms, 9])

    dofs[:, :3] = Ms[:, :3, 3]  # translation
    dofs[:, 3:6] = 0  # rotational "delta"

    cys = numpy.sqrt(Ms[:, 0, 0] * Ms[:, 0, 0] + Ms[:, 1, 0] * Ms[:, 1, 0])

    problemSelector = (cys <= 4 * numpy.finfo(float).eps)

    dofs[~problemSelector, 6] = numpy.arctan2(
        Ms[~problemSelector, 2, 1], Ms[~problemSelector, 2, 2]
    )
    dofs[~problemSelector, 7] = numpy.arctan2(
        -Ms[~problemSelector, 2, 0], cys[~problemSelector]
    )
    dofs[~problemSelector, 8] = numpy.arctan2(
        Ms[~problemSelector, 1, 0], Ms[~problemSelector, 0, 0]
    )

    dofs[problemSelector, 6] = numpy.arctan2(
        -Ms[problemSelector, 1, 2], Ms[problemSelector, 1, 1]
    )
    dofs[problemSelector, 7] = numpy.arctan2(
        -Ms[problemSelector, 2, 0], cys[problemSelector]
    )
    dofs[problemSelector, 8] = 0.0

    return dofs


def JumpDerivatives(dofs, Ms, Mparents, f1s, f2s):
    """compute JUMP derivatives from f1/f2"""
    # trans dofs
    njumpatoms = dofs.shape[0]
    dsc_ddofs = numpy.zeros([njumpatoms, 6])
    x_axes = Mparents[:, 0:3, 0]
    y_axes = Mparents[:, 0:3, 1]
    z_axes = Mparents[:, 0:3, 2]
    dsc_ddofs[:, 0] = numpy.einsum('ij, ij->i', x_axes, f2s)
    dsc_ddofs[:, 1] = numpy.einsum('ij, ij->i', y_axes, f2s)
    dsc_ddofs[:, 2] = numpy.einsum('ij, ij->i', z_axes, f2s)

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

    dsc_ddofs[:, 3] = (
        numpy.einsum('ij, ij->i', rotdof1_axes, f1s) +
        numpy.einsum('ij, ij->i', numpy.cross(rotdof1_axes, end_pos), f2s)
    )
    dsc_ddofs[:, 4] = (
        numpy.einsum('ij, ij->i', rotdof2_axes, f1s) +
        numpy.einsum('ij, ij->i', numpy.cross(rotdof2_axes, end_pos), f2s)
    )
    dsc_ddofs[:, 5] = (
        numpy.einsum('ij, ij->i', rotdof3_axes, f1s) +
        numpy.einsum('ij, ij->i', numpy.cross(rotdof3_axes, end_pos), f2s)
    )

    return dsc_ddofs


def BondTransforms(dofs):
    """BOND dofs -> HTs"""
    natoms = dofs.shape[0]

    cp = numpy.cos(dofs[:, 2])
    sp = numpy.sin(dofs[:, 2])
    ct = numpy.cos(dofs[:, 1])
    st = numpy.sin(dofs[:, 1])
    d = dofs[:, 0]

    Ms = numpy.zeros([natoms, 4, 4])
    Ms[:, 0, 0] = ct
    Ms[:, 0, 1] = -st
    Ms[:, 0, 3] = d * ct
    Ms[:, 1, 0] = cp * st
    Ms[:, 1, 1] = cp * ct
    Ms[:, 1, 2] = -sp
    Ms[:, 1, 3] = d * cp * st
    Ms[:, 2, 0] = sp * st
    Ms[:, 2, 1] = sp * ct
    Ms[:, 2, 2] = cp
    Ms[:, 2, 3] = d * sp * st
    Ms[:, 3, 3] = 1

    return Ms


def InvBondTransforms(Ms):
    """HTs -> BOND dofs"""
    nbondatoms = Ms.shape[0]

    dofs = numpy.empty([nbondatoms, 3])
    dofs[:, 0] = numpy.sqrt(numpy.square(Ms[:, :3, 3]).sum(axis=1))
    dofs[:, 1] = numpy.arctan2(-Ms[:, 0, 1], Ms[:, 0, 0])
    dofs[:, 2] = numpy.arctan2(-Ms[:, 1, 2], Ms[:, 2, 2])

    return dofs


def BondDerivatives(dofs, Ms, Mparents, f1s, f2s):
    """compute BOND derivatives from f1/f2"""
    nbondatoms = dofs.shape[0]

    end_pos = Mparents[:, 0:3, 3]
    phi_axes = Mparents[:, 0:3, 0]
    theta_axes = Ms[:, 0:3, 2]
    d_axes = Ms[:, 0:3, 0]

    dsc_ddofs = numpy.zeros([nbondatoms, 3])

    dsc_ddofs[:, 0] = numpy.einsum('ij, ij->i', d_axes, f2s)
    dsc_ddofs[:, 1] = (
        -numpy.sign(dofs[:, 1]) * (
            numpy.einsum('ij, ij->i', theta_axes, f1s) +
            numpy.einsum('ij, ij->i', numpy.cross(theta_axes, end_pos), f2s)
        )
    )
    dsc_ddofs[:, 2] = (
        -numpy.einsum('ij, ij->i', phi_axes, f1s) +
        -numpy.einsum('ij, ij->i', numpy.cross(phi_axes, end_pos), f2s)
    )

    return dsc_ddofs


def HTs_from_frames(Cs, Xs, Ys, Zs):
    """xyzs -> HTs"""
    natoms = Cs.shape[0]

    Ms = numpy.zeros([natoms, 4, 4])

    Ms[:, :3, 0] = Xs - Ys
    Ms[:, :3, 0] = (
        Ms[:, :3, 0] /
        numpy.sqrt(numpy.square(Ms[:, :3, 0]).sum(axis=1)[:, numpy.newaxis])
    )
    Ms[:, :3, 2] = numpy.cross(Ms[:, :3, 0], Zs - Xs)
    Ms[:, :3, 2] = (
        Ms[:, :3, 2] /
        numpy.sqrt(numpy.square(Ms[:, :3, 2]).sum(axis=1)[:, numpy.newaxis])
    )
    Ms[:, :3, 1] = numpy.cross(Ms[:, :3, 2], Ms[:, :3, 0])
    Ms[:, :3, 3] = Cs
    Ms[:, 3, 3] = 1

    return (Ms)


def backwardKin(kintree, coords):
    """xyzs -> HTs, dofs

      - "backward" kinematics
    """
    natoms = coords.shape[0]

    parents = kintree["parent"]
    frames = kintree["frame"]

    # 1) global HTs
    HTs = HTs_from_frames(
        coords, coords[frames[:, 0], :], coords[frames[:, 1], :],
        coords[frames[:, 2], :]
    )

    # 2) local HTs
    localHTs = numpy.empty([natoms, 4, 4])
    localHTs[1:] = numpy.matmul(HTinv(HTs[parents[1:], :, :]), HTs[1:, :, :])

    # 3) dofs
    dofs = numpy.zeros([natoms, 9])

    bondSelector = (kintree["doftype"] == BOND)
    bondSelector[0] = False
    dofs[bondSelector, :3] = InvBondTransforms(localHTs[bondSelector, :3])

    jumpSelector = (kintree["doftype"] == JUMP)
    jumpSelector[0] = False
    dofs[jumpSelector, :9] = InvJumpTransforms(localHTs[jumpSelector, :9])

    return (HTs, dofs)


def forwardKin(kintree, dofs):
    """dofs -> HTs, xyzs

      - "forward" kinematics
    """
    natoms = dofs.shape[0]

    parents = kintree["parent"]

    # 1) local HTs
    HTs = numpy.empty([natoms, 4, 4])

    bondSelector = (kintree["doftype"] == BOND)
    HTs[bondSelector, :, :] = BondTransforms(dofs[bondSelector, 0:3])

    jumpSelector = (kintree["doftype"] == JUMP)
    HTs[jumpSelector, :, :] = JumpTransforms(dofs[jumpSelector, 0:9])

    # 2) global HTs (rewrite 1->N in-place)
    SegScan(HTs, parents, HTcollect)

    coords = numpy.zeros([natoms, 3])
    coords = numpy.matmul(HTs, [0, 0, 0, 1])[:, :3]
    return (HTs, coords)


def resolveDerivs(kintree, dofs, HTs, dsc_dx):
    """xyz derivs -> dof derivs

    - derivative mapping using Abe and Go approach
    """

    natoms = dofs.shape[0]

    parents = kintree["parent"]

    # 1) local f1/f2s
    Xs = HTs[:, 0:3, 3]
    f1s = numpy.cross(Xs, Xs - dsc_dx)
    f2s = dsc_dx

    # 2) pass f1/f2s up tree
    SegScan(f1s, parents, Fscollect)
    SegScan(f2s, parents, Fscollect)

    # 3) convert to dscore/dtors
    dsc_ddofs = numpy.zeros([natoms, 9])
    bondSelector = (kintree["doftype"] == BOND)
    dsc_ddofs[bondSelector, 0:3] = BondDerivatives(
        dofs[bondSelector, :],
        HTs[bondSelector, :, :],
        HTs[parents[bondSelector], :, :],
        f1s[bondSelector, :],
        f2s[bondSelector, :],
    )
    jumpSelector = (kintree["doftype"] == JUMP)
    dsc_ddofs[jumpSelector, 0:6] = JumpDerivatives(
        dofs[jumpSelector, :],
        HTs[jumpSelector, :, :],
        HTs[parents[jumpSelector], :, :],
        f1s[jumpSelector, :],
        f2s[jumpSelector, :],
    )

    return dsc_ddofs


def writePDB(kintree, coords):
    """debugging: dump a PDB-like file"""
    atom_record_format = "ATOM  {:5d} {:^4}{:^1}{:3s} {:1}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}"
    for i in numpy.arange(len(kintree)):
        print(
            atom_record_format.format(
                i + 1,
                kintree["atom_name"][i],
                " ",
                "ALA",
                "A",
                kintree["resnum"][i],
                " ",
                coords[i, 0],
                coords[i, 1],
                coords[i, 2],
                1,
                0,
            )
        )
