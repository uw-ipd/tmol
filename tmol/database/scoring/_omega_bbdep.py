import attr
import torch

from typing import Tuple

from tmol.types import Tensor

from ._content_hash import content_hash


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
    # content-derived; caches key on it, so it must change with the contents
    uniq_id: str
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
            db = torch.load(fname, mmap=True)
        return attr.evolve(db, uniq_id=db.content_id())

    def content_id(self) -> str:
        """Identity derived from the lookup and the tables themselves."""
        return content_hash(self.bbdep_omega_lookup, self.bbdep_omega_tables)
