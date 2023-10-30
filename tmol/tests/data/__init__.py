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
def pertuzumab_pdb():
    # Pertuzumab is an antibody that binds to a protein
    # called "ERBB2." The co-crystal struction is 1s78.
    # This (truncated) PDB consists of chains A, C, and
    # D where chains C and D are the antibody (pertuzumab)
    # and chain A is the antigen (Erbb2). To retrieve
    # only the pertuzumab sequence, return the subset
    # of the PDB file starting at line 4278 (with 81
    # characters per line)
    return pdb.data["1s78.pdb"][4278 * 81 :]


@pytest.fixture()
def erbb2_and_pertuzumab_pdb():
    return pdb.data["1s78.pdb"]


@pytest.fixture()
def pertuzumab_and_nearby_erbb2_pdb_and_segments():
    # Return two things that are needed for construction of a special
    # kind of Pose that contains two complete chains (the pertuzumab
    # antibody) and then a subset of the residues in the antigen chain
    # (the ERBB2 protein) that are in close proximity to pertuzumab.
    # 1. the lines from the 1s78 PDB containing the necessary atom
    # records for the two chains and the 8 segments of ERBB2, and
    # 2. a numpy array indicating which residues should not be treated
    # as forming a chemical bond to their i-1 or i+1 neighbors; this
    # array will need to be converted to a torch tensor before being
    # given to the "pose_stack_from_canonical_form" function.
    import numpy

    # res-res     line-line
    # 127-129 3    924- 945
    # 154-156 3   1151-1176
    # 234-236 3   1724-1753
    # 244-258 15  1804-1924
    # 267-273 7   1984-2035
    # 283-290 8   2107-2158
    # 293-298 6   2173-2221
    # 309-317 9   2297-2360

    pert_lines = pdb.data["1s78.pdb"]

    def line_range(s, e):
        return pert_lines[(s - 1) * 81 : (e - 1) * 81]

    # first, give pertuzumab and
    pert_and_erbb2_lines = "".join(
        [
            pert_lines[4278 * 81 :],
            line_range(924, 945),
            line_range(1151, 1176),
            line_range(1724, 1753),
            line_range(1804, 1924),
            line_range(1984, 2035),
            line_range(2107, 2158),
            line_range(2173, 2221),
            line_range(2297, 2360),
        ]
    )

    segment_lengths = (214, 222, 3, 3, 3, 15, 7, 8, 6, 9)

    seg_range_end = numpy.cumsum(numpy.array(segment_lengths, dtype=numpy.int32))
    seg_range_start = numpy.concatenate(
        (numpy.zeros((1,), dtype=numpy.int32), seg_range_end[:-1])
    )
    n_res_tot = seg_range_end[-1]
    res_not_connected = numpy.zeros((1, n_res_tot, 2), dtype=bool)
    # do not make any of the ERBB2 residues n- or c-termini,
    # and also do not connect residues that are both part of that chain
    # that span gaps
    res_not_connected[0, seg_range_start[2:], 0] = True
    res_not_connected[0, seg_range_end[2:] - 1, 1] = True

    return (pert_and_erbb2_lines, res_not_connected)
