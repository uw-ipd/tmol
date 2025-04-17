import pytest
import numpy
from tmol.kinematics.fold_forest import EdgeType


@pytest.fixture
def ff_2ubq_6res_H():
    max_n_edges = 6
    ff_edges = numpy.full(
        (2, max_n_edges, 4),
        -1,
        dtype=numpy.int32,
    )
    ff_edges[0, 0, 0] = EdgeType.polymer
    ff_edges[0, 0, 1] = 1
    ff_edges[0, 0, 2] = 0

    ff_edges[0, 1, 0] = EdgeType.polymer
    ff_edges[0, 1, 1] = 1
    ff_edges[0, 1, 2] = 2

    ff_edges[0, 2, 0] = EdgeType.jump
    ff_edges[0, 2, 1] = 1
    ff_edges[0, 2, 2] = 4
    ff_edges[0, 2, 3] = 0

    ff_edges[0, 3, 0] = EdgeType.polymer
    ff_edges[0, 3, 1] = 4
    ff_edges[0, 3, 2] = 3

    ff_edges[0, 4, 0] = EdgeType.polymer
    ff_edges[0, 4, 1] = 4
    ff_edges[0, 4, 2] = 5

    ff_edges[0, 5, 0] = EdgeType.root_jump
    ff_edges[0, 5, 1] = -1
    ff_edges[0, 5, 2] = 1

    # Let's flip the jump and root the tree at res 4
    ff_edges[1, 0, 0] = EdgeType.polymer
    ff_edges[1, 0, 1] = 1
    ff_edges[1, 0, 2] = 0

    ff_edges[1, 1, 0] = EdgeType.polymer
    ff_edges[1, 1, 1] = 1
    ff_edges[1, 1, 2] = 2

    ff_edges[1, 2, 0] = EdgeType.jump
    ff_edges[1, 2, 1] = 4
    ff_edges[1, 2, 2] = 1
    ff_edges[1, 2, 3] = 0

    ff_edges[1, 3, 0] = EdgeType.polymer
    ff_edges[1, 3, 1] = 4
    ff_edges[1, 3, 2] = 3

    ff_edges[1, 4, 0] = EdgeType.polymer
    ff_edges[1, 4, 1] = 4
    ff_edges[1, 4, 2] = 5

    ff_edges[1, 5, 0] = EdgeType.root_jump
    ff_edges[1, 5, 1] = -1
    ff_edges[1, 5, 2] = 4

    return ff_edges


@pytest.fixture
def ff_3_jagged_ubq_465res_H():
    max_n_edges = 6
    ff_edges = numpy.full(
        (3, max_n_edges, 4),
        -1,
        dtype=numpy.int32,
    )
    # 4 res pose
    ff_edges[0, 0, 0] = EdgeType.polymer
    ff_edges[0, 0, 1] = 1
    ff_edges[0, 0, 2] = 0

    ff_edges[0, 1, 0] = EdgeType.polymer
    ff_edges[0, 1, 1] = 1
    ff_edges[0, 1, 2] = 2

    ff_edges[0, 2, 0] = EdgeType.jump
    ff_edges[0, 2, 1] = 1
    ff_edges[0, 2, 2] = 3
    ff_edges[0, 2, 3] = 0

    ff_edges[0, 3, 0] = EdgeType.root_jump
    ff_edges[0, 3, 1] = -1
    ff_edges[0, 3, 2] = 1

    # 6 res pose
    ff_edges[1, 0, 0] = EdgeType.polymer
    ff_edges[1, 0, 1] = 1
    ff_edges[1, 0, 2] = 0

    ff_edges[1, 1, 0] = EdgeType.polymer
    ff_edges[1, 1, 1] = 1
    ff_edges[1, 1, 2] = 2

    ff_edges[1, 2, 0] = EdgeType.jump
    ff_edges[1, 2, 1] = 4
    ff_edges[1, 2, 2] = 1
    ff_edges[1, 2, 3] = 0

    ff_edges[1, 3, 0] = EdgeType.polymer
    ff_edges[1, 3, 1] = 4
    ff_edges[1, 3, 2] = 3

    ff_edges[1, 4, 0] = EdgeType.polymer
    ff_edges[1, 4, 1] = 4
    ff_edges[1, 4, 2] = 5

    ff_edges[1, 5, 0] = EdgeType.root_jump
    ff_edges[1, 5, 1] = -1
    ff_edges[1, 5, 2] = 4

    # 5 res Pose
    ff_edges[2, 0, 0] = EdgeType.polymer
    ff_edges[2, 0, 1] = 1
    ff_edges[2, 0, 2] = 0

    ff_edges[2, 1, 0] = EdgeType.polymer
    ff_edges[2, 1, 1] = 1
    ff_edges[2, 1, 2] = 2

    ff_edges[2, 2, 0] = EdgeType.jump
    ff_edges[2, 2, 1] = 4
    ff_edges[2, 2, 2] = 1
    ff_edges[2, 2, 3] = 0

    ff_edges[2, 3, 0] = EdgeType.polymer
    ff_edges[2, 3, 1] = 4
    ff_edges[2, 3, 2] = 3

    ff_edges[2, 4, 0] = EdgeType.root_jump
    ff_edges[2, 4, 1] = -1
    ff_edges[2, 4, 2] = 4

    return ff_edges


@pytest.fixture
def ff_3_jagged_ubq_465res_star():
    max_n_edges = 6
    ff_edges = numpy.full(
        (3, max_n_edges, 4),
        -1,
        dtype=numpy.int32,
    )
    for i, nres in enumerate([4, 6, 5]):
        for j in range(nres):
            ff_edges[i, j, 0] = EdgeType.root_jump
            ff_edges[i, j, 1] = -1
            ff_edges[i, j, 2] = j

    return ff_edges


@pytest.fixture
def ff_2ubq_6res_U():
    max_n_edges = 4
    ff_edges_cpu = numpy.full(
        (2, max_n_edges, 4),
        -1,
        dtype=numpy.int32,
    )
    ff_edges_cpu[0, 0, 0] = EdgeType.polymer
    ff_edges_cpu[0, 0, 1] = 2
    ff_edges_cpu[0, 0, 2] = 0

    ff_edges_cpu[0, 1, 0] = EdgeType.jump
    ff_edges_cpu[0, 1, 1] = 2
    ff_edges_cpu[0, 1, 2] = 5
    ff_edges_cpu[0, 1, 3] = 0

    ff_edges_cpu[0, 2, 0] = EdgeType.polymer
    ff_edges_cpu[0, 2, 1] = 5
    ff_edges_cpu[0, 2, 2] = 3

    ff_edges_cpu[0, 3, 0] = EdgeType.root_jump
    ff_edges_cpu[0, 3, 1] = -1
    ff_edges_cpu[0, 3, 2] = 2

    # Let's flip the jump and root the tree at res 5
    ff_edges_cpu[1, 0, 0] = EdgeType.polymer
    ff_edges_cpu[1, 0, 1] = 2
    ff_edges_cpu[1, 0, 2] = 0

    ff_edges_cpu[1, 1, 0] = EdgeType.jump
    ff_edges_cpu[1, 1, 1] = 5
    ff_edges_cpu[1, 1, 2] = 2
    ff_edges_cpu[1, 1, 3] = 0

    ff_edges_cpu[1, 2, 0] = EdgeType.polymer
    ff_edges_cpu[1, 2, 1] = 5
    ff_edges_cpu[1, 2, 2] = 3

    ff_edges_cpu[1, 3, 0] = EdgeType.root_jump
    ff_edges_cpu[1, 3, 1] = -1
    ff_edges_cpu[1, 3, 2] = 5

    return ff_edges_cpu


@pytest.fixture
def ff_2ubq_6res_K():
    max_n_edges = 6
    ff_edges_cpu = numpy.full(
        (2, max_n_edges, 4),
        -1,
        dtype=numpy.int32,
    )
    ff_edges_cpu[0, 0, 0] = EdgeType.polymer
    ff_edges_cpu[0, 0, 1] = 1
    ff_edges_cpu[0, 0, 2] = 0

    ff_edges_cpu[0, 1, 0] = EdgeType.polymer
    ff_edges_cpu[0, 1, 1] = 1
    ff_edges_cpu[0, 1, 2] = 2

    ff_edges_cpu[0, 2, 0] = EdgeType.jump
    ff_edges_cpu[0, 2, 1] = 1
    ff_edges_cpu[0, 2, 2] = 3
    ff_edges_cpu[0, 2, 3] = 0

    ff_edges_cpu[0, 3, 0] = EdgeType.jump
    ff_edges_cpu[0, 3, 1] = 1
    ff_edges_cpu[0, 3, 2] = 4
    ff_edges_cpu[0, 3, 3] = 1

    ff_edges_cpu[0, 4, 0] = EdgeType.polymer
    ff_edges_cpu[0, 4, 1] = 4
    ff_edges_cpu[0, 4, 2] = 5

    ff_edges_cpu[0, 5, 0] = EdgeType.root_jump
    ff_edges_cpu[0, 5, 1] = -1
    ff_edges_cpu[0, 5, 2] = 1

    # Let's flip everything
    ff_edges_cpu[1, 0, 0] = EdgeType.polymer
    ff_edges_cpu[1, 0, 1] = 4
    ff_edges_cpu[1, 0, 2] = 3

    ff_edges_cpu[1, 1, 0] = EdgeType.polymer
    ff_edges_cpu[1, 1, 1] = 4
    ff_edges_cpu[1, 1, 2] = 5

    ff_edges_cpu[1, 2, 0] = EdgeType.jump
    ff_edges_cpu[1, 2, 1] = 4
    ff_edges_cpu[1, 2, 2] = 2
    ff_edges_cpu[1, 2, 3] = 0

    ff_edges_cpu[1, 3, 0] = EdgeType.jump
    ff_edges_cpu[1, 3, 1] = 4
    ff_edges_cpu[1, 3, 2] = 1
    ff_edges_cpu[1, 3, 3] = 1

    ff_edges_cpu[1, 4, 0] = EdgeType.polymer
    ff_edges_cpu[1, 4, 1] = 1
    ff_edges_cpu[1, 4, 2] = 0

    ff_edges_cpu[1, 5, 0] = EdgeType.root_jump
    ff_edges_cpu[1, 5, 1] = -1
    ff_edges_cpu[1, 5, 2] = 4

    return ff_edges_cpu
