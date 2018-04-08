import os
import attr

from .hbond import HBondDatabase
from .ljlk import LJLKDatabase

@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:

    ljlk : LJLKDatabase
    hbond : HBondDatabase

    @classmethod
    def from_file(cls, path=os.path.dirname(__file__)):
        return cls(
            ljlk = LJLKDatabase.from_file(os.path.join(path, "ljlk.yaml")),
            hbond = HBondDatabase.from_file(os.path.join(path, "hbond.yaml")),
        )
