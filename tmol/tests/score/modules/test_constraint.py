import pytest
import torch
import numpy

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.constraint import ConstraintScore
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystemStack

## A module implementing TR-Rosetta (and RoseTTAFold) style constraints
## (fd) this should probably be given a more specific name
DSTEP, ALPHA, MEFF, EBASE, EREP = 0.5, 1.57, 1e-4, -0.5, [10.0, 3.0, 0.5]


def preprocess_distance_constraints(dists):
    dists = torch.tensor(dists, dtype=torch.float)

    # spline bins
    xclose = torch.tensor([0.0, 2.0, 3.5])
    xmid = torch.arange(4.25, 4.25 + DSTEP * 32, DSTEP)
    xlong = torch.tensor([4.25 + DSTEP * 32, 4.25 + DSTEP * 33, 9999])
    bins = torch.cat([xclose, xmid, xlong])

    # splines
    # prob = torch.sum(dists[:, :, 5:], dim=-1)  # prob of dist within 20A
    bkgr = (xmid / 19.5) ** ALPHA
    ymid = (
        -torch.log(
            (dists[:, :, 5:] + MEFF)
            / (dists[:, :, -1][:, :, None] * bkgr[None, None, :])
        )
        + EBASE
    )
    yclose = (
        torch.max(ymid[:, :, 0], torch.tensor(0.0))[:, :, None]
        + torch.tensor(EREP)[None, None, :]
    )
    ylong = torch.zeros_like(yclose)
    dists = torch.cat([yclose, ymid, ylong], dim=-1)
    dists = torch.triu(dists, 3)

    return bins, dists


def preprocess_torsion_constraints(angles, symm):
    angles = torch.tensor(angles, dtype=torch.float)

    # spline bins
    nbins = angles.shape[2] - 1 + 4
    ASTEP = 2.0 * numpy.pi / (nbins - 4)
    bins = torch.linspace(-numpy.pi - 1.5 * ASTEP, numpy.pi + 1.5 * ASTEP, nbins)

    # splines
    prob = torch.sum(angles[:, :, 1:], dim=-1)
    if symm:
        prob = torch.triu(prob, 3)
    else:
        prob = torch.triu(prob, 3) + torch.tril(prob, 3)
    angles = -torch.log((angles + MEFF) / (angles[:, :, -1] + MEFF)[:, :, None])
    angles = torch.cat([angles[:, :, -2:], angles[:, :, 1:], angles[:, :, 1:3]], dim=-1)
    angles[prob < 0.55, :] = 0  # 0.55 = PCUT+P_ADD_OMEGA

    return bins, angles


def preprocess_angle_constraints(angles):
    angles = torch.tensor(angles, dtype=torch.float)

    nbins = angles.shape[2] - 1 + 4
    ASTEP = numpy.pi / (nbins - 4)
    bins = torch.linspace(-1.5 * ASTEP, numpy.pi + 1.5 * ASTEP, nbins)

    prob = torch.sum(angles[:, :, 1:], dim=-1)
    prob = torch.triu(prob, 3) + torch.tril(prob, 3)

    angles = -torch.log((angles + MEFF) / (angles[:, :, -1] + MEFF)[:, :, None])
    angles = torch.cat(
        [
            torch.flip(angles[:, :, 1:3], (-1,)),
            angles[:, :, 1:],
            torch.flip(angles[:, :, -2:], (-1,)),
        ],
        axis=-1,
    )
    angles[prob < 0.65, :] = 0  # 0.65 = PCUT+P_ADD_PHI

    return bins, angles


@pytest.mark.benchmark(group="score_setup")
def test_cst_score_setup(benchmark, cst_system, cst_csts, torch_device):
    @benchmark
    def score_graph():
        dists, Edists = preprocess_distance_constraints(cst_csts["dist"])
        omegas, Eomegas = preprocess_torsion_constraints(cst_csts["omega"], symm=True)
        theta, Ethetas = preprocess_torsion_constraints(cst_csts["theta"], symm=False)
        phi, Ephis = preprocess_angle_constraints(cst_csts["phi"])

        cstdata = {
            "dense_cbcb_dist_xs": dists.to(torch_device),
            "dense_cbcb_dist_ys": Edists.to(torch_device),
            "dense_cacbcbca_tors_xs": omegas.to(torch_device),
            "dense_cacbcbca_tors_ys": Eomegas.to(torch_device),
            "dense_ncacacb_tors_xs": theta.to(torch_device),
            "dense_ncacacb_tors_ys": Ethetas.to(torch_device),
            "dense_cacacb_angle_xs": phi.to(torch_device),
            "dense_cacacb_angle_ys": Ephis.to(torch_device),
        }

        return ScoreSystem.build_for(
            cst_system,
            {ConstraintScore},
            weights={"cst": 1.0},
            cstdata=cstdata,
            device=torch_device,
        )

    score_graph


def test_cst_for_system(cst_system, cst_csts, torch_device):
    dists, Edists = preprocess_distance_constraints(cst_csts["dist"])
    omegas, Eomegas = preprocess_torsion_constraints(cst_csts["omega"], symm=True)
    theta, Ethetas = preprocess_torsion_constraints(cst_csts["theta"], symm=False)
    phi, Ephis = preprocess_angle_constraints(cst_csts["phi"])

    cstdata = {
        "dense_cbcb_dist_xs": dists.to(torch_device),
        "dense_cbcb_dist_ys": Edists.to(torch_device),
        "dense_cacbcbca_tors_xs": omegas.to(torch_device),
        "dense_cacbcbca_tors_ys": Eomegas.to(torch_device),
        "dense_ncacacb_tors_xs": theta.to(torch_device),
        "dense_ncacacb_tors_ys": Ethetas.to(torch_device),
        "dense_cacacb_angle_xs": phi.to(torch_device),
        "dense_cacacb_angle_ys": Ephis.to(torch_device),
    }

    cst_score = ScoreSystem.build_for(
        cst_system,
        {ConstraintScore},
        weights={"cst_atompair": 1.0, "cst_dihedral": 1.0, "cst_angle": 1.0},
        cstdata=cstdata,
        device=torch_device,
    )

    coords = coords_for(cst_system, cst_score)
    tot = cst_score.intra_total(coords)

    torch.testing.assert_allclose(tot.cpu(), -15955.91015625)


@pytest.mark.benchmark(group="score_components")
@pytest.mark.parametrize("nstacks", [2, 3, 10, 30, 100])
def test_cst_for_stacked_system(benchmark, cst_system, cst_csts, nstacks, torch_device):
    dists, Edists = preprocess_distance_constraints(cst_csts["dist"])
    omegas, Eomegas = preprocess_torsion_constraints(cst_csts["omega"], symm=True)
    theta, Ethetas = preprocess_torsion_constraints(cst_csts["theta"], symm=False)
    phi, Ephis = preprocess_angle_constraints(cst_csts["phi"])

    cstdata = {
        "dense_cbcb_dist_xs": dists.to(torch_device),
        "dense_cbcb_dist_ys": Edists.to(torch_device),
        "dense_cacbcbca_tors_xs": omegas.to(torch_device),
        "dense_cacbcbca_tors_ys": Eomegas.to(torch_device),
        "dense_ncacacb_tors_xs": theta.to(torch_device),
        "dense_ncacacb_tors_ys": Ethetas.to(torch_device),
        "dense_cacacb_angle_xs": phi.to(torch_device),
        "dense_cacacb_angle_ys": Ephis.to(torch_device),
    }

    stack = PackedResidueSystemStack((cst_system,) * nstacks)

    stacked_score = ScoreSystem.build_for(
        stack,
        {ConstraintScore},
        weights={"cst_atompair": 1.0, "cst_dihedral": 1.0, "cst_angle": 1.0},
        cstdata=cstdata,
        device=torch_device,
    )
    coords = coords_for(stack, stacked_score)

    @benchmark
    def stack_score_constraints():
        return stacked_score.intra_total(coords)

    tot = stack_score_constraints
    torch.testing.assert_allclose(tot.cpu(), -15955.91015625 * nstacks)
