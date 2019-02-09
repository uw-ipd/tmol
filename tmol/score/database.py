from typing import Optional

from tmol.database import ParameterDatabase

from .score_graph import score_graph


@score_graph
class ParamDB:
    """Graph component containing the common database.

    Attributes:
        parameter_database: A single, shared database for all graph components.
    """

    @staticmethod
    def factory_for(val, parameter_database: Optional[ParameterDatabase] = None, **_):
        """Overridable clone-constructor.

        Initialize from ``val.parameter_database`` if possible, otherwise
        default ParameterDatabase.
        """
        if parameter_database is None:
            if getattr(val, "parameter_database", None):
                parameter_database = val.parameter_database
            else:
                parameter_database = ParameterDatabase.get_default()

        return dict(parameter_database=parameter_database)

    parameter_database: ParameterDatabase
