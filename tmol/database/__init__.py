import os
import attr

from .chemical import ChemicalDatabase
from .scoring import ScoringDatabase

# maybe this should live in the database?
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase


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
    chemical: PatchedChemicalDatabase = attr.ib()

    @classmethod
    def from_file(cls, path):
        chemdb = ChemicalDatabase.from_file(os.path.join(path, "chemical"))
        patched_chemdb = PatchedChemicalDatabase.from_chem_db(chemdb)  # apply patches
        return cls(
            scoring=ScoringDatabase.from_file(os.path.join(path, "scoring")),
            chemical=patched_chemdb,
        )
