"""NumPy-only scoring parameter records and required table loading."""

from __future__ import annotations

import os
from typing import Tuple

import attr
import numpy as np
import yaml

from ._structure import structure


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepMappingParams:
    table_id: str
    res_middle: str
    res_upper: str = "_"
    invert_phi: bool = False
    invert_psi: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepTables:
    table_id: str
    mu: np.ndarray
    sigma: np.ndarray
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _OmegaBBDepLookupRaw:
    omega_bbdep_lookup: Tuple[OmegaBBDepMappingParams, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepDatabase:
    uniq_id: str
    bbdep_omega_lookup: Tuple[OmegaBBDepMappingParams, ...]
    bbdep_omega_tables: Tuple[OmegaBBDepTables, ...] = ()

    @staticmethod
    def load_tables_from_npz(npz_path: str) -> Tuple[OmegaBBDepTables, ...]:
        """Load omega_bbdep tables from a numpy ``.npz`` archive."""
        with np.load(npz_path, allow_pickle=False) as data:
            n = int(data["__n_tables"])
            out = []
            for i in range(n):
                out.append(
                    OmegaBBDepTables(
                        table_id=str(data[f"t{i}__table_id"]),
                        mu=np.asarray(data[f"t{i}__mu"], dtype=np.float32),
                        sigma=np.asarray(data[f"t{i}__sigma"], dtype=np.float32),
                        bbstep=tuple((float(x) for x in data[f"t{i}__bbstep"])),
                        bbstart=tuple((float(x) for x in data[f"t{i}__bbstart"])),
                    )
                )
        return tuple(out)

    @classmethod
    def from_file(cls, path):
        with open(path) as stream:
            rows = structure(yaml.safe_load(stream), _OmegaBBDepLookupRaw)
        stem = os.path.splitext(path)[0]
        tables = cls.load_tables_from_npz(stem + ".npz")
        return cls(os.path.abspath(path), rows.omega_bbdep_lookup, tables)
