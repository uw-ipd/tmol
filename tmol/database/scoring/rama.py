import attr
import cattr
import numpy
import os
import re
import numpy
import yaml
import zarr

from typing import Tuple

from tmol.types.array import NDArray
from tmol.types.functional import validate_args


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaMappingParams:
    name: str
    res_middle: str
    res_upper: str = ""
    invert_phi: bool = False
    invert_psi: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaTables:
    name: str
    prob: NDArray(float)
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


def load_tables_from_zarr(path_tables):
    alltables = []
    store = zarr.ZipStore(path_tables, mode="r")
    root = zarr.Group(store)
    for aa in root:
        alltables.append(
            RamaTables(
                name=aa,
                prob=numpy.array(root[aa]["prob"]),
                bbstep=root[aa]["prob"].attrs["bbstep"],
                bbstart=root[aa]["prob"].attrs["bbstart"],
            )
        )
    return alltables


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaDatabase:
    rama_lookup: Tuple[RamaMappingParams, ...]
    rama_tables: Tuple[RamaTables, ...]

    @classmethod
    def from_files(cls, path_lookup, path_tables):
        with open(path_lookup, "r") as infile_lookup:
            raw = yaml.load(infile_lookup)
            rama_lookup = cattr.structure(
                raw["rama_lookup"], attr.fields(cls).rama_lookup.type
            )

        rama_tables = load_tables_from_zarr(path_tables)

        return cls(rama_lookup=rama_lookup, rama_tables=rama_tables)
