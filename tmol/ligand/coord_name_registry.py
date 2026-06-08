"""Process-wide ``{res_name: {coord_key: atom_name}}`` registry.

Ligand inputs (mol2/CIF) often name atoms differently from the complex PDB
they're scored in. Since both share coordinates, ``prepare_single_ligand``
records a coordinate->name map that ``pose_stack_from_biotite`` uses to remap
PDB ligand atoms onto the prepared residue type. Keys are coordinates quantized
to 1e-2 Angstrom.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

CoordKey = Tuple[int, int, int]

_LOCK = threading.Lock()
_REGISTRY: Dict[str, Dict[CoordKey, str]] = {}


def coord_key(x: float, y: float, z: float) -> CoordKey:
    """Quantize a coordinate to a hashable key (1e-2 Angstrom tolerance)."""
    return (round(x * 100.0), round(y * 100.0), round(z * 100.0))


def register_ligand_coords(
    res_name: str, coords_by_name: Dict[str, Tuple[float, float, float]]
) -> None:
    """Record a prepared ligand's ``coord_key -> atom_name`` map by res_name."""
    if not coords_by_name:
        return
    mapping: Dict[CoordKey, str] = {}
    for name, (x, y, z) in coords_by_name.items():
        mapping[coord_key(x, y, z)] = name
    with _LOCK:
        _REGISTRY[res_name] = mapping


def nearest_name(
    coord_map: Dict[CoordKey, str], x: float, y: float, z: float
) -> Optional[str]:
    """Atom name for ``(x, y, z)``, probing the 27-neighborhood of the quantized
    key to absorb rounding-boundary noise. Exact match wins; ties break by
    nearest key.
    """
    kx, ky, kz = coord_key(x, y, z)
    exact = coord_map.get((kx, ky, kz))
    if exact is not None:
        return exact
    best: Optional[str] = None
    best_d = 99
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                name = coord_map.get((kx + dx, ky + dy, kz + dz))
                if name is None:
                    continue
                d = abs(dx) + abs(dy) + abs(dz)
                if d < best_d:
                    best_d = d
                    best = name
    return best


def lookup_ligand_coords(res_name: str) -> Optional[Dict[CoordKey, str]]:
    """Return the registered ``coord_key -> atom_name`` map for ``res_name``."""
    with _LOCK:
        return _REGISTRY.get(res_name)


def clear() -> None:
    """Clear the registry (test isolation)."""
    with _LOCK:
        _REGISTRY.clear()
