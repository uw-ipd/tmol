import attr
import torch

from typing import Tuple

from tmol.types import Tensor


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
    table: Tensor[torch.float32]
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaDatabase:
    uniq_id: str  # unique id for memoization
    rama_lookup: Tuple[RamaMappingParams, ...]
    rama_tables: Tuple[RamaTables, ...]

    @classmethod
    def from_file(cls, fname: str):
        _OLD = "tmol.database.scoring.rama"
        with torch.serialization.safe_globals(
            [
                RamaDatabase,
                (RamaDatabase, f"{_OLD}.RamaDatabase"),
                RamaTables,
                (RamaTables, f"{_OLD}.RamaTables"),
                RamaMappingParams,
                (RamaMappingParams, f"{_OLD}.RamaMappingParams"),
            ]
        ):
            return torch.load(fname, mmap=True, weights_only=True)
