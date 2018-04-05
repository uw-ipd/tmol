import os
import attr

from .chemical import ChemicalDatabase
from .scoring  import ScoringDatabase

@attr.s
class ParameterDatabase:
    scoring  : ScoringDatabase = attr.ib()
    chemical : ChemicalDatabase = attr.ib()

    @classmethod
    def from_file(cls, path):
        return cls(
            scoring  = ScoringDatabase.from_file(os.path.join(path, "scoring")),
            chemical = ChemicalDatabase.from_file(os.path.join(path, "chemical"))
        )

default = ParameterDatabase.from_file(os.path.join(os.path.dirname(__file__), "default"))
