import attr
import torch
import numpy

from typing import Tuple

from tmol.types.array import NDArray


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
    table: NDArray[float]
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaDatabase:
    uniq_id: str  # unique id for memoization
    rama_lookup: Tuple[RamaMappingParams, ...]
    rama_tables: Tuple[RamaTables, ...]

    @classmethod
    def from_file(cls, fname: str):
        with torch.serialization.safe_globals(
            [
                RamaDatabase,
                RamaTables,
                RamaMappingParams,
                numpy.core.multiarray._reconstruct,
                numpy.ndarray,
                numpy.dtype,
                numpy.dtypes.Float64DType,
            ]
        ):
            return torch.load(fname, weights_only=False)
