import os
import attr

from .ljlk import LJLKDatabase


@attr.s(auto_attribs=True, slots=True, frozen=True)
class ScoringDatabase:

    ljlk: LJLKDatabase

    @classmethod
    def from_file(cls, path=os.path.dirname(__file__)):
        return cls(
            ljlk=LJLKDatabase.from_file(os.path.join(path, "ljlk.yaml"))
        )
