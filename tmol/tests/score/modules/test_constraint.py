import copy

import pytest
from pytest import approx
import torch

from tmol.database import ParameterDatabase

from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.constraint import ConstraintParameters, ConstraintScore
from tmol.score.modules.coords import coords_for

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


def preprocess_constraints(data):
    DSTEP, ALPHA, MEFF, EBASE, EREP = 0.5, 1.57, 1e-4, -0.5, [10.0, 3.0, 0.5]
    dist = torch.tensor(data["dist"], dtype=torch.float)

    # spline bins
    xclose = torch.tensor([0.0, 2.0, 3.5])
    xmid = torch.arange(4.25, 4.25 + DSTEP * 32, DSTEP)
    xlong = torch.tensor([4.25 + DSTEP * 32, 4.25 + DSTEP * 33, 9999])
    bins = torch.cat([xclose, xmid, xlong])

    # splines
    prob = torch.sum(dist[:, :, 5:], dim=-1)  # prob of dist within 20A
    bkgr = (xmid / 19.5) ** ALPHA
    ymid = (
        -torch.log(
            (dist[:, :, 5:] + MEFF) / (dist[:, :, -1][:, :, None] * bkgr[None, None, :])
        )
        + EBASE
    )
    yclose = (
        torch.max(ymid[:, :, 0], torch.tensor(0.0))[:, :, None]
        + torch.tensor(EREP)[None, None, :]
    )
    ylong = torch.zeros_like(yclose)
    dists = torch.cat([yclose, ymid, ylong], dim=-1)

    return bins, dists


@pytest.mark.benchmark(group="score_setup")
def test_cst_score_setup(benchmark, cst_system, cst_csts, torch_device):
    @benchmark
    def score_graph():
        xs, ys = preprocess_constraints(cst_csts)
        cstdata = {
            "dense_cb_spline_xs": xs.to(torch_device),
            "dense_cb_spline_ys": ys.to(torch_device),
        }
        return ScoreSystem.build_for(
            cst_system,
            {ConstraintScore},
            weights={"cst": 1.0},
            cstdata=cstdata,
            device=torch_device,
        )

    graph = score_graph


def test_cst_for_system(benchmark, benchmark_pass, cst_system, cst_csts, torch_device):
    xs, ys = preprocess_constraints(cst_csts)
    cstdata = {
        "dense_cb_spline_xs": xs.to(torch_device),
        "dense_cb_spline_ys": ys.to(torch_device),
    }

    cst_score = ScoreSystem.build_for(
        cst_system,
        {ConstraintScore},
        weights={"cst": 1.0},
        cstdata=cstdata,
        device=torch_device,
    )

    coords = coords_for(cst_system, cst_score)
    tot = cst_score.intra_total(coords)

    torch.testing.assert_allclose(tot.cpu(), -4890.0249)


def test_cst_for_stacked_system(ubq_system: PackedResidueSystem):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))

    stacked_score = ScoreSystem.build_for(
        twoubq, {ConstraintScore}, weights={"cst": 1.0}
    )
    coords = coords_for(twoubq, stacked_score)

    tot = stacked_score.intra_total(coords)
    assert tot.shape == (2,)
    torch.testing.assert_allclose(tot[0], tot[1])

    sumtot = torch.sum(tot)
    sumtot.backward()
