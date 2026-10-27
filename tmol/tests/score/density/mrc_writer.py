import gzip
import struct

import numpy


def write_mrc(
    path,
    values_xyz,
    *,
    cell=None,
    cell_grid=None,
    start=(0, 0, 0),
    axis_order=(1, 2, 3),
    origin=(0.0, 0.0, 0.0),
    symops=(),
    endian="<",
):
    """Write values [x, y, z] as an MRC/CCP4 map with the given header fields."""
    values_xyz = numpy.asarray(values_xyz, dtype=numpy.float32)
    if cell_grid is None:
        cell_grid = values_xyz.shape
    if cell is None:
        cell = (*map(float, cell_grid), 90.0, 90.0, 90.0)
    mapc, mapr, maps = axis_order
    raw = values_xyz.transpose(maps - 1, mapr - 1, mapc - 1)
    ns, nr, nc = raw.shape
    sym = b"".join(op.ljust(80).encode() for op in symops)

    header = bytearray(1024)
    struct.pack_into(f"{endian}4i", header, 0, nc, nr, ns, 2)
    struct.pack_into(
        f"{endian}3i", header, 16, start[mapc - 1], start[mapr - 1], start[maps - 1]
    )
    struct.pack_into(f"{endian}3i", header, 28, *cell_grid)
    struct.pack_into(f"{endian}6f", header, 40, *cell)
    struct.pack_into(f"{endian}3i", header, 64, *axis_order)
    struct.pack_into(f"{endian}i", header, 92, len(sym))
    struct.pack_into(f"{endian}3f", header, 196, *origin)
    header[208:212] = b"MAP "
    header[212:216] = b"DA\x00\x00" if endian == "<" else b"\x11\x11\x00\x00"
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wb") as out:
        out.write(header)
        out.write(sym)
        out.write(raw.astype(f"{endian}f4").tobytes())
