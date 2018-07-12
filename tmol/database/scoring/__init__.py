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

    @toolz.functoolz.memoize
    @validate_args
    def get_compacted_rama_db(
            self, device: torch.device
    ) -> CompactedRamaDatabase:
        return CompactedRamaDatabase.from_ramadb(self.rama, device)
