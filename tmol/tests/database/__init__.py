import pytest

import tmol.database
import tmol.database.chemical


@pytest.fixture
def default_unpatched_chemical_database():
    return tmol.database.chemical.ChemicalDatabase.get_default()


@pytest.fixture
def default_database():
    return tmol.database.ParameterDatabase.get_default()
