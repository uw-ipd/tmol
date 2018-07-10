import pytest
import torch
import tmol.score.ljlk.params as params
import tmol.database as database
import tmol.system.score_support as score_support
import tmol.score.ljlk.potentials as potentials
from tmol.tests.torch import requires_cuda

import math
import numba.cuda as cuda
from tmol.utility.numba import as_cuda_array


def test_ljlk_full(benchmark, ubq_system, torch_device):

    coords = score_support.coords_for_system(
        ubq_system, torch_device, requires_grad=False
    )["coords"]
    atom_types = (
        score_support.bonded_atoms_for_system(ubq_system)["atom_types"]
    )

    db = database.ParameterDatabase.get_default()
    param_resolver = params.LJLKParamResolver.from_database(
        db.scoring.ljlk, torch_device
    )

    ljlk_atom_pair_params = param_resolver[atom_types.reshape((-1, 1)),
                                           atom_types.reshape((1, -1))]

    gparams = param_resolver.global_params
    pparams = ljlk_atom_pair_params
    idx = torch.arange(len(atom_types), device=torch_device, dtype=torch.long)
    pidx = [idx[:, None], idx[None, :]]

    @benchmark
    def lj_total_score():
        atom_pair_dist = (coords[:, None, :] - coords[None, :, :]).norm(dim=-1)
        ljlk_interaction_weight = torch.full_like(atom_pair_dist, 1)

        return float(
            potentials.lj_score(
                # Distance
                dist=atom_pair_dist,

                # Bonded params
                interaction_weight=ljlk_interaction_weight,

                # Pair params
                lj_sigma=pparams.lj_sigma[pidx],
                lj_switch_slope=pparams.lj_switch_slope[pidx],
                lj_switch_intercept=pparams.lj_switch_intercept[pidx],
                lj_coeff_sigma12=pparams.lj_coeff_sigma12[pidx],
                lj_coeff_sigma6=pparams.lj_coeff_sigma6[pidx],
                lj_spline_y0=pparams.lj_spline_y0[pidx],
                lj_spline_dy0=pparams.lj_spline_dy0[pidx],

                # Global params
                lj_switch_dis2sigma=gparams.lj_switch_dis2sigma,
                spline_start=gparams.spline_start,
                max_dis=gparams.max_dis,
            ).sum()
        )


def test_ljlk_preindexed(benchmark, ubq_system, torch_device):

    coords = score_support.coords_for_system(
        ubq_system, torch_device, requires_grad=False
    )["coords"]
    atom_types = (
        score_support.bonded_atoms_for_system(ubq_system)["atom_types"]
    )

    db = database.ParameterDatabase.get_default()
    param_resolver = params.LJLKParamResolver.from_database(
        db.scoring.ljlk, torch_device
    )

    ljlk_atom_pair_params = param_resolver[atom_types.reshape((-1, 1)),
                                           atom_types.reshape((1, -1))]

    gparams = param_resolver.global_params
    pparams = ljlk_atom_pair_params
    idx = torch.arange(len(atom_types), device=torch_device, dtype=torch.long)
    pidx = [idx[:, None], idx[None, :]]

    atom_pair_dist = (coords[:, None, :] - coords[None, :, :]).norm(dim=-1)
    aparams = dict(
        interaction_weight=torch.full_like(atom_pair_dist, 1),

        # Pair params
        lj_sigma=pparams.lj_sigma[pidx],
        lj_switch_slope=pparams.lj_switch_slope[pidx],
        lj_switch_intercept=pparams.lj_switch_intercept[pidx],
        lj_coeff_sigma12=pparams.lj_coeff_sigma12[pidx],
        lj_coeff_sigma6=pparams.lj_coeff_sigma6[pidx],
        lj_spline_y0=pparams.lj_spline_y0[pidx],
        lj_spline_dy0=pparams.lj_spline_dy0[pidx],

        # Global params
        lj_switch_dis2sigma=gparams.lj_switch_dis2sigma,
        spline_start=gparams.spline_start,
        max_dis=gparams.max_dis,
    )

    @benchmark
    def lj_total_score():
        atom_pair_dist = (coords[:, None, :] - coords[None, :, :]).norm(dim=-1)

        return float(
            potentials.lj_score(
                # Distance
                dist=atom_pair_dist,
                **aparams
            ).sum()
        )


def test_ljlk_dist(benchmark, ubq_system, torch_device):

    coords = score_support.coords_for_system(
        ubq_system, torch_device, requires_grad=False
    )["coords"]

    @benchmark
    def lj_total_score():
        return (coords[:, None, :] - coords[None, :, :]).norm(dim=-1)[0]


def test_ljlk_predist(benchmark, ubq_system, torch_device):

    coords = score_support.coords_for_system(
        ubq_system, torch_device, requires_grad=False
    )["coords"]
    atom_types = (
        score_support.bonded_atoms_for_system(ubq_system)["atom_types"]
    )

    db = database.ParameterDatabase.get_default()
    param_resolver = params.LJLKParamResolver.from_database(
        db.scoring.ljlk, torch_device
    )

    ljlk_atom_pair_params = param_resolver[atom_types.reshape((-1, 1)),
                                           atom_types.reshape((1, -1))]

    gparams = param_resolver.global_params
    pparams = ljlk_atom_pair_params
    idx = torch.arange(len(atom_types), device=torch_device, dtype=torch.long)
    pidx = [idx[:, None], idx[None, :]]

    atom_pair_dist = (coords[:, None, :] - coords[None, :, :]).norm(dim=-1)
    aparams = dict(
        interaction_weight=torch.full_like(atom_pair_dist, 1),

        # Pair params
        lj_sigma=pparams.lj_sigma[pidx],
        lj_switch_slope=pparams.lj_switch_slope[pidx],
        lj_switch_intercept=pparams.lj_switch_intercept[pidx],
        lj_coeff_sigma12=pparams.lj_coeff_sigma12[pidx],
        lj_coeff_sigma6=pparams.lj_coeff_sigma6[pidx],
        lj_spline_y0=pparams.lj_spline_y0[pidx],
        lj_spline_dy0=pparams.lj_spline_dy0[pidx],

        # Global params
        lj_switch_dis2sigma=gparams.lj_switch_dis2sigma,
        spline_start=gparams.spline_start,
        max_dis=gparams.max_dis,
    )

    @benchmark
    def lj_total_score():
        return float(
            potentials.lj_score(
                # Distance
                dist=atom_pair_dist,
                **aparams
            ).sum()
        )


@cuda.jit(device=True)
def _dist(A, B):
    return math.sqrt(A[0] * B[0] + A[1] * B[1] + A[2] * B[2])


@cuda.jit
def _cudist(A, B, D):
    x, y = cuda.grid(2)

    if x < A.shape[0] and y < B.shape[0]:
        D[x, y] = _dist(A[x], B[y])


def cudist(x, y, r):
    assert x.shape[-1] == 3
    assert y.shape[-1] == 3
    assert r.shape == (x.shape[0], y.shape[0])
    tpb = (8, 8)
    bpg = (math.ceil(x.shape[0] / 8), math.ceil(y.shape[0] / 8))

    _cudist[bpg, tpb](x, y, r)

    return r


@requires_cuda
@pytest.mark.parametrize("factor", [1, 2, 3, 4, 5])
def test_ljlk_cudist_ten(benchmark, ubq_system, factor):
    torch_device = torch.device("cuda")

    coords = score_support.coords_for_system(
        ubq_system, torch_device, requires_grad=False
    )["coords"]

    coords = torch.cat([coords] * factor, dim=0)

    result = coords.new_empty((len(coords), len(coords)))
    r = as_cuda_array(result[:, :])

    ca = as_cuda_array(coords)
    cb = as_cuda_array(coords)

    @benchmark
    def lj_total_score():
        cudist(ca, cb, r)
        cuda.synchronize()
