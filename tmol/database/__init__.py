import os
import attr

from .chemical import ChemicalDatabase

@attr.s
class ParameterDatabase:
    chemical : ChemicalDatabase = attr.ib()

    @classmethod
    def from_file(cls, path):
        return cls(
            chemical = ChemicalDatabase.from_file(os.path.join(path, "chemical"))
        )

default = ParameterDatabase.from_file(os.path.join(os.path.dirname(__file__), "default"))
