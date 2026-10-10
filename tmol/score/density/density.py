"""Density maps: the map container and an MRC/CCP4 reader."""

from __future__ import annotations

import gzip
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy
import torch


@dataclass(frozen=True)
class ElectronDensityMap:
    """A density map on a (possibly skewed) regular grid.

    Attributes:
        density: Map values with shape [z, y, x].
        origin: Cartesian coordinate of voxel [0, 0, 0], in [x, y, z].
        voxel_basis: [3, 3] matrix whose column i is the Cartesian step along
            grid axis i (x, y, z); diagonal for an orthogonal grid.
        resolution: Map resolution in A; map files do not record it.
    """

    density: torch.Tensor
    origin: torch.Tensor
    voxel_basis: torch.Tensor
    resolution: float

    def to(self, *args, **kwargs) -> "ElectronDensityMap":
        """Return a copy whose tensors have been passed through Tensor.to."""
        return ElectronDensityMap(
            self.density.to(*args, **kwargs),
            self.origin.to(*args, **kwargs),
            self.voxel_basis.to(*args, **kwargs),
            self.resolution,
        )


def frac_to_cart(cell: tuple[float, ...]) -> numpy.ndarray:
    """Fractional-to-Cartesian matrix of a unit cell (a, b, c, alpha, beta, gamma)."""
    a, b, c = cell[:3]
    ca, cb, cg = (math.cos(math.radians(x)) for x in cell[3:])
    sg = math.sin(math.radians(cell[5]))
    sb = math.sin(math.radians(cell[4]))
    return numpy.array(
        [
            [a, b * cg, c * cb],
            [0.0, b * sg, c * (ca - cb * cg) / sg],
            [0.0, 0.0, c * sb * math.sqrt(1.0 - ((cb * cg - ca) / (sb * sg)) ** 2)],
        ]
    )


_SYMOP_TERM = re.compile(r"([+-]?)(\d+/\d+|\d*\.\d+|\d+|[XYZ])")


def parse_symops(text: str) -> list[tuple[numpy.ndarray, numpy.ndarray]]:
    """Parse CCP4 symmetry operators ("-X,Y+1/2,-Z") into fractional (R, t).

    Returns an empty list if any record is not a symmetry operator.
    """
    ops = []
    records = [r.strip() for line in text.splitlines() for r in line.split("*")]
    for record in records:
        if not record:
            continue
        rows = record.upper().replace(" ", "").split(",")
        if len(rows) != 3:
            return []
        rot, trans = numpy.zeros((3, 3)), numpy.zeros(3)
        for i, row in enumerate(rows):
            if not row or "".join(m.group(0) for m in _SYMOP_TERM.finditer(row)) != row:
                return []
            for sign, term in _SYMOP_TERM.findall(row):
                s = -1.0 if sign == "-" else 1.0
                if term in "XYZ":
                    rot[i, "XYZ".index(term)] += s
                elif "/" in term:
                    num, den = term.split("/")
                    trans[i] += s * float(num) / float(den)
                else:
                    trans[i] += s * float(term)
        ops.append((rot, trans))
    return ops


def expand_to_unit_cell(
    box: numpy.ndarray,
    origin_index: numpy.ndarray,
    cell_grid: tuple[int, int, int],
    symops: list[tuple[numpy.ndarray, numpy.ndarray]],
) -> numpy.ndarray:
    """Fill a whole unit cell from a sub-box using the space-group operators.

    box is [x, y, z]; origin_index is the grid index of box[0, 0, 0]. Voxel t
    of the result sits at grid index origin_index + t. Voxels with no symmetry
    mate inside the box are zero.
    """
    m = numpy.asarray(cell_grid)
    n = numpy.asarray(box.shape)
    cell = numpy.zeros(tuple(m), dtype=box.dtype)
    cell[: n[0], : n[1], : n[2]] = box
    missing = numpy.ones(tuple(m), dtype=bool)
    missing[: n[0], : n[1], : n[2]] = False
    target = numpy.argwhere(missing)
    frac = (target + origin_index) / m
    for rot, trans in symops:
        if not len(target):
            break
        mate = frac @ rot.T + trans
        src = numpy.mod(numpy.floor(mate * m + 0.5 - origin_index), m).astype(int)
        inside = numpy.all(src < n, axis=1)
        t, s = target[inside], src[inside]
        cell[t[:, 0], t[:, 1], t[:, 2]] = box[s[:, 0], s[:, 1], s[:, 2]]
        target, frac = target[~inside], frac[~inside]
    return cell


def _open_binary(path: str | Path) -> BinaryIO:
    path = Path(path)
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _mrc_endian(header: bytes) -> str:
    """Infer MRC byte order from dimensions, mode, and the machine stamp."""
    little = struct.unpack_from("<4i", header, 0)
    big = struct.unpack_from(">4i", header, 0)

    def plausible(values):
        return all(0 < n < 10_000_000 for n in values[:3]) and values[3] in {
            0,
            1,
            2,
            6,
        }

    if plausible(little) and not plausible(big):
        return "<"
    if plausible(big) and not plausible(little):
        return ">"
    # 0x44 0x41 is the standard little-endian IEEE machine stamp.
    return "<" if header[212:214] in (b"DA", b"DD", b"\x44\x41") else ">"


def read_mrc(
    path: str | Path,
    resolution: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> ElectronDensityMap:
    """Read an MRC/CCP4 density map of the given resolution, including gzipped maps.

    Handles axis permutations (MAPC/MAPR/MAPS), skewed cells, CCP4 symmetry
    records, the MRC2014 Cartesian origin, and start-index origins. A map that
    covers less than the unit cell (MX/MY/MZ larger than the extent) is expanded
    to the full cell with its symmetry operators, as Rosetta does.
    """
    with _open_binary(path) as inp:
        header = inp.read(1024)
        if len(header) != 1024:
            raise ValueError(f"{path}: truncated MRC header")
        if header[208:211] != b"MAP":
            raise ValueError(f"{path}: missing MRC MAP signature")

        endian = _mrc_endian(header)
        nc, nr, ns, mode = struct.unpack_from(f"{endian}4i", header, 0)
        starts = struct.unpack_from(f"{endian}3i", header, 16)
        cell_grid = list(struct.unpack_from(f"{endian}3i", header, 28))
        cell = struct.unpack_from(f"{endian}6f", header, 40)
        mapc, mapr, maps = struct.unpack_from(f"{endian}3i", header, 64)
        nsymbt = struct.unpack_from(f"{endian}i", header, 92)[0]
        cart_origin = struct.unpack_from(f"{endian}3f", header, 196)
        rest = inp.read()

    if (mapc, mapr, maps) == (0, 0, 0):
        mapc, mapr, maps = 1, 2, 3
    if sorted((mapc, mapr, maps)) != [1, 2, 3]:
        raise ValueError(f"{path}: invalid MRC axis order {(mapc, mapr, maps)}")
    if min(cell[:3]) <= 0:
        raise ValueError(f"{path}: invalid unit cell {cell}")

    mode_dtype = {
        0: numpy.dtype("i1"),
        1: numpy.dtype(f"{endian}i2"),
        2: numpy.dtype(f"{endian}f4"),
        6: numpy.dtype(f"{endian}u2"),
    }.get(mode)
    if mode_dtype is None:
        raise ValueError(f"{path}: unsupported MRC mode {mode}")

    count = nc * nr * ns
    n_bytes = count * mode_dtype.itemsize
    if nsymbt < 0 or len(rest) < nsymbt + n_bytes:
        if len(rest) == n_bytes:
            nsymbt = 0  # bogus symmetry record size
        else:
            raise ValueError(f"{path}: truncated MRC density payload")
    # symmetry records are 80-character lines without newlines
    sym = rest[:nsymbt].decode("ascii", errors="replace")
    sym_text = "\n".join(sym[i : i + 80] for i in range(0, len(sym), 80))
    payload = rest[nsymbt : nsymbt + n_bytes]

    raw = numpy.frombuffer(payload, dtype=mode_dtype, count=count).reshape(ns, nr, nc)
    # raw axes are (section, row, column); reorder to Cartesian [x, y, z]
    raw_axis_for_cart = {maps: 0, mapr: 1, mapc: 2}
    xyz = raw.transpose(
        raw_axis_for_cart[1], raw_axis_for_cart[2], raw_axis_for_cart[3]
    ).astype(numpy.float32)
    extent = xyz.shape

    start_xyz = numpy.zeros(3)
    for axis, start in zip((mapc, mapr, maps), starts, strict=True):
        start_xyz[axis - 1] = start
    for i in range(3):
        if cell_grid[i] <= 0:
            cell_grid[i] = extent[i]

    f2c = frac_to_cart(cell)
    voxel_basis = f2c / numpy.asarray(cell_grid, dtype=numpy.float64)
    # as Rosetta: the MRC2014 origin is used only if all three components are set
    if all(o != 0 and abs(o) < 1e4 for o in cart_origin):
        origin = numpy.asarray(cart_origin, dtype=numpy.float64)
        origin_index = numpy.linalg.solve(voxel_basis, origin)
    else:
        origin_index = start_xyz
        origin = voxel_basis @ origin_index

    if all(m >= n for m, n in zip(cell_grid, extent, strict=True)) and any(
        m != n for m, n in zip(cell_grid, extent, strict=True)
    ):
        symops = parse_symops(sym_text) or parse_symops("X,Y,Z")
        xyz = expand_to_unit_cell(xyz, origin_index, tuple(cell_grid), symops)

    zyx = numpy.ascontiguousarray(xyz.transpose(2, 1, 0))
    return ElectronDensityMap(
        density=torch.as_tensor(zyx, dtype=dtype, device=device),
        origin=torch.as_tensor(origin, dtype=dtype, device=device),
        voxel_basis=torch.as_tensor(voxel_basis, dtype=dtype, device=device),
        resolution=float(resolution),
    )


def block_type_atomic_numbers(packed_block_types, device=None) -> torch.Tensor:
    """Atomic number of every atom of every block type, [n_block_types, max_n_atoms].

    Padding atoms and virtual atoms are zero.
    """
    pbt = packed_block_types
    element_for_type = {t.name: t.element for t in pbt.chem_db.atom_types}
    z_for_element = {e.name: e.atomic_number for e in pbt.chem_db.element_types}
    z = numpy.zeros((pbt.n_types, pbt.max_n_atoms), dtype=numpy.int64)
    for i, block_type in enumerate(pbt.active_block_types):
        for j, atom in enumerate(block_type.atoms):
            z[i, j] = z_for_element[element_for_type[atom.atom_type]]
    return torch.as_tensor(z, device=device)
