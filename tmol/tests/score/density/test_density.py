import gzip
import struct

import numpy
import pytest
import torch

from tmol.score.density import DensityCorrelation, ElectronDensityMap, read_mrc


def _small_map(dtype=torch.float64):
    return ElectronDensityMap(
        torch.zeros((13, 13, 13), dtype=dtype),
        torch.tensor((-6.0, -6.0, -6.0), dtype=dtype),
        torch.ones(3, dtype=dtype),
    )


def test_self_correlation_and_coordinate_gradient():
    template = _small_map()
    renderer = DensityCorrelation(template, resolution=2.5, atom_chunk_size=2)
    atomic_numbers = torch.tensor([6, 7, 8], dtype=torch.int64)
    reference = torch.tensor(
        [[-1.2, 0.1, 0.0], [1.1, -0.3, 0.4], [0.2, 1.4, -0.7]],
        dtype=torch.float64,
    )
    observed = renderer.synthesize_density(reference, atomic_numbers).detach()
    scorer = DensityCorrelation(
        ElectronDensityMap(observed, template.origin, template.voxel_size),
        resolution=2.5,
        atom_chunk_size=2,
    )

    assert scorer.correlation(reference, atomic_numbers).item() == pytest.approx(
        1.0, abs=1e-10
    )
    shifted = (reference + torch.tensor([1.0, 0.0, 0.0])).requires_grad_(True)
    energy = scorer(shifted, atomic_numbers)
    energy.backward()
    assert energy.item() > -0.99
    assert torch.isfinite(shifted.grad).all()
    assert shifted.grad.abs().sum() > 0


def test_batch_and_padding_atoms():
    density_map = _small_map(dtype=torch.float32)
    scorer = DensityCorrelation(density_map, resolution=3.0)
    coords = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
            [[1.0, 0.0, 0.0], [9.0, 9.0, 9.0]],
        ]
    )
    atomic_numbers = torch.tensor([[6, 0], [6, 1]])
    rendered = scorer.synthesize_density(coords, atomic_numbers)
    assert rendered.shape == (2, 13, 13, 13)
    assert rendered[0].sum() == pytest.approx(rendered[1].sum(), rel=1e-5)


def _write_mrc(path, values, axis_order=(1, 2, 3), origin=(1.5, 2.5, 3.5)):
    nz, ny, nx = values.shape
    value_axis_for_cart = {3: 0, 2: 1, 1: 2}
    mapc, mapr, maps = axis_order
    raw = values.transpose(
        value_axis_for_cart[maps],
        value_axis_for_cart[mapr],
        value_axis_for_cart[mapc],
    )
    ns, nr, nc = raw.shape
    header = bytearray(1024)
    struct.pack_into("<4i", header, 0, nc, nr, ns, 2)
    struct.pack_into("<3i", header, 28, nx, ny, nz)
    struct.pack_into("<6f", header, 40, nx * 0.5, ny * 0.75, nz, 90, 90, 90)
    struct.pack_into("<3i", header, 64, *axis_order)
    struct.pack_into("<3f", header, 196, *origin)
    header[208:212] = b"MAP "
    header[212:216] = b"DA\x00\x00"
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wb") as out:
        out.write(header)
        out.write(raw.astype("<f4").tobytes())


@pytest.mark.parametrize("suffix", [".mrc", ".map.gz"])
def test_read_mrc_axis_order_and_gzip(tmp_path, suffix):
    values = numpy.arange(2 * 3 * 4, dtype=numpy.float32).reshape(2, 3, 4)
    path = tmp_path / ("test" + suffix)
    _write_mrc(path, values, axis_order=(2, 3, 1))
    result = read_mrc(path)
    assert torch.equal(result.density, torch.from_numpy(values))
    assert torch.allclose(result.origin, torch.tensor([1.5, 2.5, 3.5]))
    assert torch.allclose(result.voxel_size, torch.tensor([0.5, 0.75, 1.0]))
