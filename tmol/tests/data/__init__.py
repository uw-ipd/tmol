import pytest
from . import pdb
from . import rosetta_baseline


@pytest.fixture(scope="session")
def ubq_pdb():
    return pdb.data["1ubq"]


@pytest.fixture(scope="session")
def ubq_res():
    from tmol.system.residue.io import ResidueReader

    return ResidueReader().parse_pdb(pdb.data["1ubq"])


@pytest.fixture()
def ubq_system():
    from tmol.system.residue.io import read_pdb

    return read_pdb(pdb.data["1ubq"])


@pytest.fixture(scope="session")
def ubq_rosetta_baseline():
    # TODO ubq baseline does *not* contain the same conformation as ubq_pdb
    return rosetta_baseline.data["1ubq"]
