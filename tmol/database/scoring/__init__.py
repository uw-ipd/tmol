import os
import attr

from .hbond import HBondDatabase
from .ljlk import LJLKDatabase
from .rama import RamaDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:

    ljlk: LJLKDatabase
    hbond: HBondDatabase
    rama: RamaDatabase

    @classmethod
    def from_file(cls, path=os.path.dirname(__file__)):  # noqa
        return cls(
            ljlk=LJLKDatabase.from_file(os.path.join(path, "ljlk.yaml")),
            hbond=HBondDatabase.from_file(os.path.join(path, "hbond.yaml")),
            rama=RamaDatabase.from_file(os.path.join(path, "rama.json")),
        )
