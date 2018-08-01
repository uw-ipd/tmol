import attr
import torch
import numpy

from tmol.types.functional import validate_args

from tmol.types.torch import Tensor

from .params import (WaterBuildingParams)

Params = Tensor(torch.float)[...]
CoordArray = Tensor(torch.double)[:, 3]
WatersArray = Tensor(torch.double)[..., 3]
BoolArray = Tensor(torch.uint8)[:]


# get lk energy, 1-sided (replicates LK code)
@validate_args
def get_lk_1way(
        # dist d->desolv
        dist: Params,

        # Pair score parameters (one sided)
        lj_rad: Params,
        lk_coeff: Params,
        lk_inv_lambda2: Params,
        lk_spline_close_dy1: Params,
        lk_spline_close_x0: Params,
        lk_spline_close_x1: Params,
        lk_spline_close_y0: Params,
        lk_spline_close_y1: Params,
        lk_spline_far_dy0: Params,
        lk_spline_far_y0: Params,
        spline_start: Params,
        max_dis: Params
):
    real = dist.dtype
    invdist2 = 1 / (dist * dist)

    flat_selector = (dist < lk_spline_close_x0)
    flat_component = lk_spline_close_y0

    near_spline_selector = ((dist >= lk_spline_close_x0) &
                            (dist < lk_spline_close_x1))
    x = dist
    x0 = lk_spline_close_x0
    x1 = lk_spline_close_x1
    y0 = lk_spline_close_y0
    y1 = lk_spline_close_y1
    dy1 = lk_spline_close_dy1
    u0 = (3.0 / (x1 - x0)) * ((y1 - y0) / (x1 - x0))
    u1 = (3.0 / (x1 - x0)) * (dy1 - (y1 - y0) / (x1 - x0))
    near_spline_component = (
        (x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3 * y0)
    ) / (3 * (x0 - x1))  # yapf: disable

    # analytic LK part
    analytic_selector = ((dist >= lk_spline_close_x1) & (dist < spline_start))

    dis = dist - lj_rad
    x = dis * dis * lk_inv_lambda2
    analytic_component = invdist2 * (torch.exp(-x1) * lk_coeff1)

    x0 = spline_start
    x1 = max_dis
    far_spline_selector = ((dist >= x0) & (dist < x1))
    x = dist
    y0 = lk_spline_far_y0
    dy0 = lk_spline_far_dy0
    u0 = (3.0 / (x1 - x0)) * ((-y0) / (x1 - x0) - dy0)
    u1 = (3.0 / (x1 - x0)) * (y0 / (x1 - x0))
    far_spline_component = (
        (x - x1) * ((x - x0) * (u1 * (x0 - x) + u0 * (x - x1)) + 3 * y0)
    ) / (3 * (x0 - x1))  # yapf: disable

    raw_lk = (
        flat_component * flat_selector.to(real) +
        near_spline_component * near_spline_selector.to(real) +
        analytic_component * analytic_selector.to(real) +
        far_spline_component * far_spline_selector.to(real)
    )

    return raw_lk


# build an acceptor water on base atoms with given d/angle/theta
@validate_args
def build_acc_waters(
        a: CoordArray, b: CoordArray, b0: CoordArray, dist: Params,
        angle: Params, theta: Params
) -> CoordArray:
    natoms = a.shape[0]

    def unit_norm(v):
        return v / torch.norm(v, dim=-1, keepdim=True)

    # a-b-b0 triple to coordinate frame
    Ms = torch.eye(
        4, dtype=a.dtype, device=a.device
    ).unsqueeze(0).repeat(natoms, 1, 1)
    xaxis = Ms[:, :3, 0]
    yaxis = Ms[:, :3, 1]
    zaxis = Ms[:, :3, 2]
    center = Ms[:, :3, 3]

    xaxis[:] = unit_norm(a - b)
    zaxis[:] = unit_norm(torch.cross(xaxis, b0 - a))
    yaxis[:] = unit_norm(torch.cross(zaxis, xaxis))
    center[:] = a

    # transform to matrix
    cph = torch.cos(theta)
    sph = torch.sin(theta)
    cth = torch.cos(angle)
    sth = torch.sin(angle)

    Xforms = torch.empty([natoms, 4, 4], dtype=a.dtype, device=a.device)
    Xforms[:, 0, 0] = cth
    Xforms[:, 0, 1] = -sth
    Xforms[:, 0, 2] = 0
    Xforms[:, 0, 3] = dist * cth
    Xforms[:, 1, 0] = cph * sth
    Xforms[:, 1, 1] = cph * cth
    Xforms[:, 1, 2] = -sph
    Xforms[:, 1, 3] = dist * cph * sth
    Xforms[:, 2, 0] = sph * sth
    Xforms[:, 2, 1] = sph * cth
    Xforms[:, 2, 2] = cph
    Xforms[:, 2, 3] = dist * sph * sth
    Xforms[:, 3, 0] = 0
    Xforms[:, 3, 1] = 0
    Xforms[:, 3, 2] = 0
    Xforms[:, 3, 3] = 1

    waters = torch.matmul(Ms, Xforms)[:, :3, 3]
    return waters


# build a donor water on base atoms with given d/angle/theta
@validate_args
def build_don_waters(d: CoordArray, h: CoordArray, dist: Params) -> CoordArray:
    dhn = (d - h)
    dhn = dhn / dhn.norm(dim=-1).unsqueeze(dim=-1)
    waters = d + dist.double() * dhn

    return waters


# given a collection of polar atoms, calculate an N x 6 x 3 array of water molecules
# needs to properly:
# a) handle cases where a heavyatom is both a donor and acceptor:
#    in these cases, donor and acceptor waters both need to be generated and assigned
#    to the corresponding heavyatom
# b) handle cases where a heavyatom may be donating through > 1 H
#
@validate_args
def render_waters(
        heavyatoms: CoordArray, acc_base: CoordArray, acc_base0: CoordArray,
        don_H: WatersArray, is_sp2_acceptor: BoolArray,
        is_sp3_acceptor: BoolArray, is_ring_acceptor: BoolArray,
        waterparams: WaterBuildingParams
) -> WatersArray:
    npolar = heavyatoms.shape[0]
    nacc = waterparams.max_acc_wat
    ndon = 4

    # expand donors
    # we simply expand each donor atom along up to 4 D-H vectors
    heavyatoms_exp = heavyatoms.unsqueeze(1).repeat([1, ndon, 1])
    don_wats = build_don_waters(
        heavyatoms_exp.view(-1, 3), don_H.view(-1, 3), waterparams.dist_donor
    ).view(-1, ndon, 3)

    # expand acceptors
    #  we build stacks of (dist,angle,tors) sets based on the donor type
    #  each acceptor may have >1 (dist,angle,torsion) set
    acc_wats = torch.full([npolar, nacc, 3],
                          float('nan'),
                          dtype=heavyatoms.dtype,
                          device=heavyatoms.device)
    acc_dists = torch.full([npolar, nacc],
                           float('nan'),
                           dtype=torch.float,
                           device=heavyatoms.device)
    acc_angles = acc_dists.clone()
    acc_tors = acc_dists.clone()

    if (torch.sum(is_sp2_acceptor)):
        nsp2acc_wats = len(waterparams.dists_sp2)
        acc_dists[is_sp2_acceptor, :nsp2acc_wats] = waterparams.dists_sp2
        acc_angles[is_sp2_acceptor, :nsp2acc_wats] = waterparams.angles_sp2
        acc_tors[is_sp2_acceptor, :nsp2acc_wats] = waterparams.tors_sp2

    if (torch.sum(is_sp3_acceptor)):
        nsp3acc_wats = len(waterparams.dists_sp3)
        acc_dists[is_sp2_acceptor, :nsp3acc_wats] = waterparams.dists_sp3
        acc_angles[is_sp2_acceptor, :nsp3acc_wats] = waterparams.angles_sp3
        acc_tors[is_sp2_acceptor, :nsp3acc_wats] = waterparams.tors_sp3

    if (torch.sum(is_ring_acceptor)):
        nringacc_wats = len(waterparams.dists_ring)
        acc_dists[is_ring_acceptor, :nringacc_wats] = waterparams.dists_ring
        acc_angles[is_ring_acceptor, :nringacc_wats] = waterparams.angles_ring
        acc_tors[is_ring_acceptor, :nringacc_wats] = waterparams.tors_ring

    is_acc = is_sp2_acceptor | is_sp3_acceptor | is_ring_acceptor
    if (torch.sum(is_acc)):
        heavyatoms_exp = heavyatoms[is_acc, :].unsqueeze(1).repeat([
            1, nacc, 1
        ]).view(-1, 3)
        acc_base_exp = acc_base[is_acc, :].unsqueeze(1).repeat([1, nacc, 1]
                                                               ).view(-1, 3)
        acc_base0_exp = acc_base0[is_acc, :].unsqueeze(1).repeat([1, nacc, 1]
                                                                 ).view(-1, 3)
        acc_wats[is_acc, :, :] = build_acc_waters(
            heavyatoms_exp, acc_base_exp, acc_base0_exp, acc_dists.view(-1),
            acc_angles.view(-1), acc_tors.view(-1)
        ).view(-1, nacc, 3)

    # merge arrays
    waters_out = torch.cat((don_wats, acc_wats), dim=1)

    return waters_out
