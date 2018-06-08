import pytest

import tmol.database


@pytest.fixture
def default_database():
    return tmol.database.ParameterDatabase.get_default()
