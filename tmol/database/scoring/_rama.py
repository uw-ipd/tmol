import attr
import torch

from typing import Tuple

from tmol.types import Tensor

from ._content_hash import content_hash


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
    # content-derived; caches key on it, so it must change with the contents
    uniq_id: str
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
            db = torch.load(fname, mmap=True)
        return attr.evolve(db, uniq_id=db.content_id())

    def content_id(self) -> str:
        """Identity derived from the lookup and the tables themselves."""
        return content_hash(self.rama_lookup, self.rama_tables)
