import attr
import torch
import numpy

from typing import Tuple

from tmol.types.array import NDArray


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
    mu: NDArray[float]
    sigma: NDArray[float]
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepDatabase:
    uniq_id: str  # unique id for memoization
    bbdep_omega_lookup: Tuple[OmegaBBDepMappingParams, ...]
    bbdep_omega_tables: Tuple[OmegaBBDepTables, ...]

    @classmethod
    def from_file(cls, fname: str):
        with torch.serialization.safe_globals(
            [
                OmegaBBDepDatabase,
                OmegaBBDepMappingParams,
                OmegaBBDepTables,
                numpy.core.multiarray._reconstruct,
                numpy.ndarray,
                numpy.dtype,
                numpy.dtypes.Float64DType,
            ]
        ):
            return torch.load(fname)
