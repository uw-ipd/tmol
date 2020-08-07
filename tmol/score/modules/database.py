import attr
from attrs_strict import type_validator
from typing import Optional
from functools import singledispatch

from tmol.score.modules.bases import ScoreModule, ScoreSystem

from tmol.database import ParameterDatabase


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class ParamDB(ScoreModule):
    """Score module specifying the common database.

    Attributes:
        parameter_database: A single, shared database for all graph components.
    """

    @classmethod
    def depends_on(cls):
        return set()

    @staticmethod
    @singledispatch
    def build_for(
        val,
        system: ScoreSystem,
        *,
        parameter_database: Optional[ParameterDatabase] = None,
        **_,
    ) -> "ParamDB":
        """Default constructor.

        Initialize from ``val.parameter_database`` if possible, otherwise
        default ParameterDatabase.
        """
        if parameter_database is None:
            parameter_database = ParameterDatabase.get_default()

        return ParamDB(system=system, parameter_database=parameter_database)

    parameter_database: ParameterDatabase = attr.ib(validator=type_validator())


@ParamDB.build_for.register(ScoreSystem)
def _clone_for_score_system(
    old, system, *, parameter_database: Optional[ParameterDatabase] = None, **_
) -> "ParamDB":
    """Clone-constructor for score system, default to source parameter database."""
    if parameter_database is None:
        parameter_database = ParamDB.get(old).parameter_database

    return ParamDB(system=system, parameter_database=parameter_database)
