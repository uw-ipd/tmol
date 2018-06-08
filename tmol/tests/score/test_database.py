import copy

from tmol.database import ParameterDatabase
from tmol.score.database import ParamDB


def test_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default())

    # Parameter database defaults
    src = ParamDB()
    assert src.parameter_database is ParameterDatabase.get_default()

    src: ParamDB = ParamDB.build_for(ubq_system)
    assert src.parameter_database is ParameterDatabase.get_default()

    # Parameter database is overridden via kwarg
    src: ParamDB = ParamDB.build_for(ubq_system, parameter_database=clone_db)
    assert src.parameter_database is clone_db

    # Parameter database is referenced on clone
    clone: ParamDB = ParamDB.build_for(src)
    assert clone.parameter_database is src.parameter_database

    # Parameter database is overriden on clone via kwarg
    clone: ParamDB = ParamDB.build_for(
        src, parameter_database=ParameterDatabase.get_default()
    )
    assert clone.parameter_database is not src.parameter_database
    assert clone.parameter_database is ParameterDatabase.get_default()
