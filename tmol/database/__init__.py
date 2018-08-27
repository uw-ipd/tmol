import os
import attr

from .chemical import ChemicalDatabase
from .scoring import ScoringDatabase


@attr.s
class ParameterDatabase:
    __default = None

    @classmethod
    def get_default(cls) -> "ParameterDatabase":
        """Load and return default parameter database."""
        if cls.__default is None:
            cls.__default = ParameterDatabase.from_file(
                os.path.join(os.path.dirname(__file__), "default")
            )
        return cls.__default

    scoring: ScoringDatabase = attr.ib()
    chemical: ChemicalDatabase = attr.ib()

    @classmethod
    def from_file(cls, path):
        return cls(
            scoring=ScoringDatabase.from_file(os.path.join(path, "scoring")),
            chemical=ChemicalDatabase.from_file(os.path.join(path, "chemical")),
        )
