import pytest

import tmol.system.restypes


@pytest.fixture
def default_restype_set():
    return tmol.system.restypes.ResidueTypeSet.get_default()
