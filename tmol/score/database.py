import attr

from tmol.utility.reactive import reactive_attrs

from tmol.database import ParameterDatabase


@reactive_attrs(auto_attribs=True)
class ParamDB:
    # The target torch device
    parameter_database: ParameterDatabase = attr.ib()

    @parameter_database.default
    def __default_database(self):
        return ParameterDatabase.get_default()
