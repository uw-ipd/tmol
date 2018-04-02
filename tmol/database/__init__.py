import os
import properties

from .chemical import ChemicalDatabase
from .scoring  import ScoringDatabase

class ParameterDatabase(properties.HasProperties):
    scoring  : ScoringDatabase  = properties.Instance("scoring databases", ScoringDatabase)
    chemical : ChemicalDatabase = properties.Instance("chemical composition", ChemicalDatabase)

    @classmethod
    def load(cls, path=os.path.dirname(__file__)):
        return cls(
            scoring  = ScoringDatabase.load(os.path.join(path, "scoring")),
            chemical = ChemicalDatabase(source = os.path.join(path, "basic"))
        )

default = ParameterDatabase.load()
