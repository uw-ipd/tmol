import attr
import torch

from typing import Tuple

from tmol.types import Tensor


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
    mu: Tensor[torch.float32]
    sigma: Tensor[torch.float32]
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepDatabase:
    uniq_id: str  # unique id for memoization
    bbdep_omega_lookup: Tuple[OmegaBBDepMappingParams, ...]
    bbdep_omega_tables: Tuple[OmegaBBDepTables, ...]

    @classmethod
    def from_file(cls, fname: str):
        _OLD = "tmol.database.scoring.omega_bbdep"
        with torch.serialization.safe_globals(
            [
                OmegaBBDepDatabase,
                (OmegaBBDepDatabase, f"{_OLD}.OmegaBBDepDatabase"),
                OmegaBBDepMappingParams,
                (OmegaBBDepMappingParams, f"{_OLD}.OmegaBBDepMappingParams"),
                OmegaBBDepTables,
                (OmegaBBDepTables, f"{_OLD}.OmegaBBDepTables"),
            ]
        ):
            return torch.load(fname, mmap=True, weights_only=True)
