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


@pytest.fixture(scope="session")
def disulfide_res():
    from tmol.system.io import ResidueReader

    return ResidueReader.get_default().parse_pdb(pdb.data["3plc"])


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


@pytest.fixture()
def water_box_res():
    from tmol.system.io import ResidueReader

    return ResidueReader.get_default().parse_pdb(pdb.data["water_box"])


@pytest.fixture()
def water_box_system():
    from tmol.system.io import read_pdb

    ret = read_pdb(pdb.data["water_box"])
    return ret


@pytest.fixture()
def pertuzumab_lines():
    # this PDB consists of chains A, C, and D where
    # chains C and D are the antibody (pertuzumab)
    # and chain A is the antigen (Erbb2) so to retrieve
    # only the pertuzumab sequence, return the subset
    # of the PDB file starting at line 4278 (with 81
    # characters per line)
    return pdb.data["1s78.pdb"][4278 * 81 :]


@pytest.fixture()
def erbb2_and_pertuzumab_lines():
    return pdb.data["1s78.pdb"]


@pytest.fixture()
def pert_and_nearby_erbb2():
    # res-res line-line
    # 127-129 3    924- 945
    # 154-156 3   1151-1175
    # 234-236 3   1724-1753
    # 244-258 15  1804-1924
    # 267-273 7   1984-2035
    # 283-290 8   2107-2158
    # 293-298 6   2173-2221
    # 309-317 9   2297-2360

    pert_lines = pdb.data["1s78.pdb"]

    def line_range(s, e):
        return pert_lines[(s - 1) * 81 : (e - 1) * 81]

    pert_and_erbb2_lines = "".join(
        [
            pert_lines[4278 * 81 :],
            line_range(924, 945),
            line_range(1151, 1175),
            line_range(1724, 1753),
            line_range(1804, 1924),
            line_range(1984, 2035),
            line_range(2107, 2158),
            line_range(2173, 2221),
            line_range(2297, 2360),
        ]
    )

    segment_lengths = (214, 222, 3, 3, 3, 15, 7, 8, 6, 9)

    return (pert_and_erbb2_lines, segment_lengths)
