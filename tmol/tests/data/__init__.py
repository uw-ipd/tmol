import pytest
import os

from . import pdb
from . import rosetta_baseline


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


@pytest.fixture()
def cst_system():
    from tmol.system.io import read_pdb

    return read_pdb(pdb.data["6DMZ_A.pdb"])


@pytest.fixture()
def cst_csts():
    from numpy import load

    return dict(load(os.path.dirname(__file__) + "/constraints/6DMZ_A.npz"))


@pytest.fixture(scope="session")
def ubq_rosetta_baseline():
    # TODO ubq baseline does *not* contain the same conformation as ubq_pdb
    return rosetta_baseline.data["1ubq"]


@pytest.fixture(scope="session")
def systems_bysize():
    from tmol.system.io import read_pdb

    return {
        40: read_pdb(pdb.data["bysize_040_res_5uoi.pdb"]),
        75: read_pdb(pdb.data["bysize_075_res_2mtq.pdb"]),
        150: read_pdb(pdb.data["bysize_150_res_5yzf.pdb"]),
        300: read_pdb(pdb.data["bysize_300_res_6f8b.pdb"]),
        600: read_pdb(pdb.data["bysize_600_res_5m4a.pdb"]),
    }


@pytest.fixture(scope="session")
def res_bysize():
    from tmol.system.io import ResidueReader

    return {
        40: ResidueReader.get_default().parse_pdb(pdb.data["bysize_040_res_5uoi.pdb"]),
        75: ResidueReader.get_default().parse_pdb(pdb.data["bysize_075_res_2mtq.pdb"]),
        150: ResidueReader.get_default().parse_pdb(pdb.data["bysize_150_res_5yzf.pdb"]),
        300: ResidueReader.get_default().parse_pdb(pdb.data["bysize_300_res_6f8b.pdb"]),
        600: ResidueReader.get_default().parse_pdb(pdb.data["bysize_600_res_5m4a.pdb"]),
    }


@pytest.fixture()
def water_box_res():
    from tmol.system.io import ResidueReader

    return ResidueReader.get_default().parse_pdb(pdb.data["water_box"])


@pytest.fixture()
def water_box_system():
    from tmol.system.io import read_pdb

    ret = read_pdb(pdb.data["water_box"])
    return ret
