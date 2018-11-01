import numpy
import torch
import pytest

from tmol.database.scoring.rama import RamaDatabase
from tmol.database import ParameterDatabase
from tmol.score.rama.rama_splines import RamaSplines


def test_rama_splines(torch_device):
    ramadb = RamaDatabase.from_file("tmol/database/default/scoring/rama/rama.bin")
    rsplines = RamaSplines.from_ramadb(ramadb, torch_device)
    assert rsplines.table.shape == (40, 36, 36)
    phi_vals = (
        torch.arange(36, device=torch_device)
        .reshape(-1, 1)
        .repeat(1, 36)
        .reshape(-1, 1)
    )
    psi_vals = torch.arange(36, device=torch_device).repeat(1, 36).reshape(-1, 1)
    x = torch.cat((phi_vals, psi_vals), dim=1)
    y = torch.full((36 * 36, 1), 0, dtype=torch.long, device=torch_device)
    for i in range(20):
        for j in range(2):
            y[:, 0] = i + 20 * j
            xlong = x.type(torch.long)
            inds = y[:, 0] * 36 * 36 + xlong[:, 0] * 36 + xlong[:, 1]
            original_vals = rsplines.table.reshape(-1)[inds]
            interp_vals = rsplines.bspline.interpolate(x, y)
            numpy.testing.assert_allclose(
                interp_vals.cpu().detach().numpy(),
                original_vals.cpu().detach().numpy(),
                atol=1e-5,
                rtol=1e-7,
            )


def test_load_rama_splines_once(torch_device):
    db = ParameterDatabase.get_default()
    crama1 = RamaSplines.from_ramadb(db.scoring.rama, torch_device)
    crama2 = RamaSplines.from_ramadb(db.scoring.rama, torch_device)
    assert crama1 is crama2
