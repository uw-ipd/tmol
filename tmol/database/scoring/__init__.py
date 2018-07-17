import os
import attr
import toolz.functoolz
import torch

from .hbond import HBondDatabase
from .ljlk import LJLKDatabase
from .rama import (RamaDatabase, CompactedRamaDatabase)
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:

    ljlk: LJLKDatabase
    hbond: HBondDatabase
    rama: RamaDatabase

    @classmethod
    def from_file(cls, path=os.path.dirname(__file__)):
        return cls(
            ljlk=LJLKDatabase.from_file(os.path.join(path, "ljlk.yaml")),
            hbond=HBondDatabase.from_file(os.path.join(path, "hbond.yaml")),
            rama=RamaDatabase.from_file(os.path.join(path, "rama.json")),
        )

    @validate_args
    def get_compacted_rama_db(
            self, device: torch.device
    ) -> CompactedRamaDatabase:
        """Ensure only one compacted copy of the database is created for either
        the CPU or the GPU by using a memoization of the device and the RamaDatabase;
        The RamaDatabase is hashed based on the name of the file that was used
        to create it.
        """
        return compacted_rama_db(self.rama, device)


@validate_args
@toolz.functoolz.memoize(
    key=lambda args, kwargs: (args[0], args[1].type, args[1].index)
)
def compacted_rama_db(
        ramadb: RamaDatabase, device: torch.device
) -> CompactedRamaDatabase:
    return CompactedRamaDatabase.from_ramadb(ramadb, device)
