"""NumPy-only scoring parameter records and required table loading."""

from __future__ import annotations

import os
from typing import Tuple

import attr
import numpy as np
import yaml

from ._structure import structure


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaMappingParams:
    table_id: str
    res_middle: str
    res_upper: str = "_"
    invert_phi: bool = False
    invert_psi: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaTables:
    table_id: str
    table: np.ndarray
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _RamaLookupRaw:
    rama_lookup: Tuple[RamaMappingParams, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaDatabase:
    uniq_id: str
    rama_lookup: Tuple[RamaMappingParams, ...]
    rama_tables: Tuple[RamaTables, ...] = ()

    @staticmethod
    def load_tables_from_npz(npz_path: str) -> Tuple[RamaTables, ...]:
        """Load rama energy tables from a numpy ``.npz`` archive.

        Uses the flat per-table layout written by
        ``scripts/convert_binary_tables.py``.
        """
        with np.load(npz_path, allow_pickle=False) as data:
            n = int(data["__n_tables"])
            out = []
            for i in range(n):
                out.append(
                    RamaTables(
                        table_id=str(data[f"t{i}__table_id"]),
                        table=np.asarray(data[f"t{i}__table"], dtype=np.float32),
                        bbstep=tuple((float(x) for x in data[f"t{i}__bbstep"])),
                        bbstart=tuple((float(x) for x in data[f"t{i}__bbstart"])),
                    )
                )
        return tuple(out)

    @classmethod
    def from_file(cls, path):
        with open(path) as stream:
            rows = structure(yaml.safe_load(stream), _RamaLookupRaw)
        stem = os.path.splitext(path)[0]
        tables = cls.load_tables_from_npz(stem + ".npz")
        return cls(os.path.abspath(path), rows.rama_lookup, tables)
