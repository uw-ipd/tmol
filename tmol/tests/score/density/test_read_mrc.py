import gzip

import numpy
import pytest
import torch

from tmol.score.density import read_mrc
from tmol.score.density.density import frac_to_cart, parse_symops
from tmol.tests.data import data_path

from .conftest import EM_MAP, EM_RESOLUTION, XTAL_MAP, XTAL_RESOLUTION, load_map
from .mrc_writer import write_mrc

EM_CELL = (74.86079406738281, 75.94573211669922, 83.5403060913086)
EM_START = (-37, -34, -40)
XTAL_CELL = (30.18, 38.40, 53.32, 90.0, 105.85, 90.0)
XTAL_GRID = (60, 76, 104)


def test_cryoem_header():
    em = load_map(EM_MAP)
    assert em.density.shape == (77, 70, 69)
    voxel = torch.tensor(EM_CELL, dtype=torch.float64) / torch.tensor([69, 70, 77])
    torch.testing.assert_close(em.voxel_basis, torch.diag(voxel), rtol=1e-6, atol=1e-12)
    torch.testing.assert_close(
        em.origin,
        voxel * torch.tensor(EM_START, dtype=torch.float64),
        rtol=1e-6,
        atol=0,
    )


@pytest.mark.parametrize("suffix", [".mrc", ".map.gz"])
@pytest.mark.parametrize("axis_order", [(1, 2, 3), (2, 3, 1), (3, 1, 2)])
@pytest.mark.parametrize("endian", ["<", ">"])
@pytest.mark.parametrize("use_origin_field", [False, True])
def test_cryoem_rewritten_layouts_read_the_same(
    tmp_path, suffix, axis_order, endian, use_origin_field
):
    """The real map, re-encoded with another layout, reads back unchanged."""
    em = load_map(EM_MAP)
    path = tmp_path / ("map" + suffix)
    write_mrc(
        path,
        em.density.permute(2, 1, 0).numpy(),
        cell=(*EM_CELL, 90.0, 90.0, 90.0),
        start=(0, 0, 0) if use_origin_field else EM_START,
        origin=tuple(em.origin.tolist()) if use_origin_field else (0.0, 0.0, 0.0),
        axis_order=axis_order,
        endian=endian,
    )
    result = read_mrc(path, EM_RESOLUTION, dtype=torch.float64)
    assert torch.equal(result.density, em.density)
    torch.testing.assert_close(result.origin, em.origin, rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(result.voxel_basis, em.voxel_basis)


def test_origin_field_needs_all_three_components(tmp_path):
    """As in Rosetta, an origin with a zero component is ignored for nstart."""
    em = load_map(EM_MAP)
    path = tmp_path / "map.mrc"
    write_mrc(
        path,
        em.density.permute(2, 1, 0).numpy(),
        cell=(*EM_CELL, 90.0, 90.0, 90.0),
        start=EM_START,
        origin=(0.0, 12.5, -3.0),
    )
    result = read_mrc(path, EM_RESOLUTION, dtype=torch.float64)
    torch.testing.assert_close(result.origin, em.origin)


def test_bogus_symmetry_record_size(tmp_path):
    data = bytearray(gzip.open(data_path("density", EM_MAP)).read())
    data[92:96] = (240).to_bytes(4, "little")
    path = tmp_path / "map.mrc"
    path.write_bytes(bytes(data))
    assert torch.equal(
        read_mrc(path, EM_RESOLUTION, dtype=torch.float64).density,
        load_map(EM_MAP).density,
    )


def test_parse_symops():
    ops = parse_symops("X,Y,Z\n-X, Y+1/2, -Z\n1/2+x,1/2-y,-z * -x,-y,z")
    assert len(ops) == 4
    rot, trans = ops[1]
    numpy.testing.assert_array_equal(rot, numpy.diag([-1.0, 1.0, -1.0]))
    numpy.testing.assert_array_equal(trans, [0.0, 0.5, 0.0])
    rot, trans = ops[2]
    numpy.testing.assert_array_equal(rot, numpy.diag([1.0, -1.0, -1.0]))
    numpy.testing.assert_array_equal(trans, [0.5, 0.5, 0.0])
    # extended headers that are not symmetry records are ignored
    assert parse_symops("FEI extended header") == []
    assert parse_symops("X,Y") == []


def test_crystal_asu_map_expands_to_a_symmetric_cell():
    """The 7RSA P21 map covers half the cell along b; the reader fills the rest."""
    xtal = load_map(XTAL_MAP)
    assert xtal.density.shape == XTAL_GRID[::-1]
    basis = frac_to_cart(XTAL_CELL) / numpy.asarray(XTAL_GRID)
    torch.testing.assert_close(
        xtal.voxel_basis, torch.from_numpy(basis), rtol=1e-6, atol=1e-6
    )
    assert torch.all(xtal.origin == 0)
    a_axis, c_axis = basis[:, 0], basis[:, 2]
    cos_beta = a_axis @ c_axis / numpy.linalg.norm(a_axis) / numpy.linalg.norm(c_axis)
    assert cos_beta == pytest.approx(numpy.cos(numpy.radians(105.85)), abs=1e-6)

    # every voxel equals its mate under -x, y+1/2, -z
    rho = xtal.density.permute(2, 1, 0)
    mate = torch.roll(rho.flip(0, 2), (1, XTAL_GRID[1] // 2, 1), dims=(0, 1, 2))
    assert torch.equal(rho, mate)
    assert rho.std() > 0.1


def test_crystal_map_without_symops_is_zero_filled(tmp_path):
    xtal = load_map(XTAL_MAP)
    rows = 38  # rows along b written by the ASU map
    box = xtal.density.permute(2, 1, 0)[:, :rows, :].numpy()
    path = tmp_path / "asu_no_symops.ccp4"
    write_mrc(path, box, cell=XTAL_CELL, cell_grid=XTAL_GRID, axis_order=(3, 1, 2))
    rho = read_mrc(path, XTAL_RESOLUTION, dtype=torch.float64).density
    rho = rho.permute(2, 1, 0)
    assert torch.equal(rho[:, :rows, :], xtal.density.permute(2, 1, 0)[:, :rows, :])
    assert torch.all(rho[:, rows:, :] == 0)
