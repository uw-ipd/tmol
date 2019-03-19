import pytest

from . import pdb
from . import rosetta_baseline
from . import structure


@pytest.fixture(scope="session")
def min_pdb():
    return pdb.data["bysize_015_res_1lu6"]


@pytest.fixture(scope="session")
def min_res():
    from tmol.system.io import ResidueReader

    return ResidueReader.get_default().parse_pdb(pdb.data["bysize_015_res_1lu6"])


@pytest.fixture(scope="session")
def min_system():
    from tmol.system.io import read_pdb

    return read_pdb(pdb.data["bysize_015_res_1lu6"])


@pytest.fixture(scope="session")
def big_pdb():
    return pdb.data["bysize_600_res_5m4a"]


@pytest.fixture(scope="session")
def big_res():
    from tmol.system.io import ResidueReader

    return ResidueReader.get_default().parse_pdb(pdb.data["bysize_600_res_5m4a"])


@pytest.fixture(scope="session")
def big_system():
    from tmol.system.io import read_pdb

    return read_pdb(pdb.data["bysize_600_res_5m4a"])


@pytest.fixture(scope="session")
def ubq_pdb():
    return pdb.data["1ubq"]


@pytest.fixture(scope="session")
def ubq_res():
    from tmol.system.io import ResidueReader

    return ResidueReader.get_default().parse_pdb(pdb.data["1ubq"])


@pytest.fixture()
def ubq_system():
    from tmol.system.io import read_pdb

    return read_pdb(pdb.data["1ubq"])


@pytest.fixture(scope="session")
def ubq_rosetta_baseline():
    # TODO ubq baseline does *not* contain the same conformation as ubq_pdb
    return rosetta_baseline.data["1ubq"]


@pytest.fixture()
def water_box_res():
    from tmol.system.io import ResidueReader

    return ResidueReader.get_default().parse_pdb(pdb.data["water_box"])


@pytest.fixture()
def water_box_system():
    from tmol.system.io import read_pdb

    return read_pdb(pdb.data["water_box"])


@pytest.fixture()
def structures_bysize():
    return {
        n: structure.TestStructure(d)
        for n, d in pdb.data.items()
        if n.startswith("BYSIZE")
    }
