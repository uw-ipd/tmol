import copy

from tmol.database import ParameterDatabase
from tmol.score.modules.bases import ScoreSystem
from tmol.score.modules.database import ParamDB


def test_database_clone_factory(ubq_system):
    clone_db = copy.copy(ParameterDatabase.get_default())

    # Parameter database defaults
    src = ScoreSystem._build_with_modules(object(), {ParamDB})
    assert ParamDB.get(src).parameter_database is ParameterDatabase.get_default()

    src = ScoreSystem._build_with_modules(ubq_system, {ParamDB})
    assert ParamDB.get(src).parameter_database is ParameterDatabase.get_default()

    # Parameter database is overridden via kwarg
    src = ScoreSystem._build_with_modules(
        ubq_system, {ParamDB}, parameter_database=clone_db
    )
    assert ParamDB.get(src).parameter_database is clone_db

    # Parameter database is referenced on clone
    clone = ScoreSystem._build_with_modules(src, {ParamDB})
    assert ParamDB.get(clone).parameter_database is clone_db
    assert ParamDB.get(clone).parameter_database is ParamDB.get(src).parameter_database

    # Parameter database is overriden on clone via kwarg
    clone = ScoreSystem._build_with_modules(
        src, {ParamDB}, parameter_database=ParameterDatabase.get_default()
    )
    assert ParamDB.get(clone).parameter_database is not clone_db
    assert (
        ParamDB.get(clone).parameter_database is not ParamDB.get(src).parameter_database
    )
    assert ParamDB.get(clone).parameter_database is ParameterDatabase.get_default()
