import numpy
from tmol.kinematics.builder import _KinematicBuilder
from tmol.kinematics.scan_ordering import get_scans


def kinforest_from_roots_and_bonds(roots, bonds):
    kfo_2_to, parents = _KinematicBuilder.bonds_to_forest(roots, bonds)
    kinforest = (
        _KinematicBuilder()
        .append_connected_components(
            to_roots=roots,
            kfo_2_to=kfo_2_to,
            to_parents_in_kfo=parents,
            to_jump_nodes=numpy.array([], dtype=numpy.int32),
        )
        .kinforest
    )
    kfo_2_to_w_root = numpy.zeros((kfo_2_to.shape[0] + 1,), dtype=numpy.int32)
    kfo_2_to_w_root[1:] = kfo_2_to + 1

    return kinforest, kfo_2_to_w_root


def test_get_scans_simple_path():
    """Create a forest that is simply a path
    0->1->2->3->4->5->6->7->8->9
    """
    bonds = numpy.zeros((18, 2), dtype=numpy.int32)
    bonds[:9, 0] = numpy.arange(9, dtype=numpy.int32)
    bonds[:9, 1] = numpy.arange(9, dtype=numpy.int32) + 1
    bonds[9:, 0] = bonds[:9, 1]
    bonds[9:, 1] = bonds[:9:, 0]
    roots = numpy.array([0], dtype=numpy.int32)

    kinforest, _ = kinforest_from_roots_and_bonds(roots, bonds)
    nodes, scans, gens = get_scans(
        kinforest.parent.cpu().numpy(), numpy.zeros((1,), dtype=numpy.int32)
    )

    nodes_gold = numpy.arange(11, dtype=numpy.int32)
    scans_gold = numpy.zeros((1,), dtype=numpy.int32)
    gens_gold = numpy.zeros((2, 2), dtype=numpy.int32)
    gens_gold[1, 0] = 11
    gens_gold[1, 1] = 1

    numpy.testing.assert_equal(nodes_gold, nodes)
    numpy.testing.assert_equal(scans_gold, scans)
    numpy.testing.assert_equal(gens_gold, gens)


def test_get_scans_two_simple_paths():
    """Create a forest that is simply two paths
    0->1->2->3->4->5->6->7->8->9,
    10->11->12->13->14->15->16->17->18->19,
    """
    bonds = numpy.zeros((36, 2), dtype=numpy.int32)
    bonds[:9, 0] = numpy.arange(9, dtype=numpy.int32)
    bonds[:9, 1] = numpy.arange(9, dtype=numpy.int32) + 1
    bonds[9:18, 0] = numpy.arange(9, dtype=numpy.int32) + 10
    bonds[9:18, 1] = numpy.arange(9, dtype=numpy.int32) + 11
    bonds[18:, 0] = bonds[:18, 1]
    bonds[18:, 1] = bonds[:18:, 0]
    roots = numpy.array([0, 10], dtype=numpy.int32)

    kinforest, kfo_2_to_w_root = kinforest_from_roots_and_bonds(roots, bonds)
    nodes, scans, gens = get_scans(
        kinforest.parent.cpu().numpy(), numpy.zeros((1,), dtype=numpy.int32)
    )

    nodes_to = kfo_2_to_w_root[nodes]

    nodes_gold = numpy.concatenate(
        (
            numpy.arange(11, dtype=numpy.int32),
            numpy.zeros((1,), dtype=numpy.int32),
            numpy.arange(10, dtype=numpy.int32) + 11,
        )
    )
    scans_gold = numpy.array([0, 11], dtype=numpy.int32)
    gens_gold = numpy.zeros((2, 2), dtype=numpy.int32)
    gens_gold[1, 0] = 22
    gens_gold[1, 1] = 2

    numpy.testing.assert_equal(nodes_gold, nodes_to)
    numpy.testing.assert_equal(scans_gold, scans)
    numpy.testing.assert_equal(gens_gold, gens)


def test_get_scans_three_simple_paths():
    """Create a forest that is simply three paths
    0->1->2->3->4->5->6->7->8->9,
    10->11->12->13->14->15->16->17->18->19,
    20->21->22->23->24->25->26->27->28->29,
    """
    bonds = numpy.zeros((54, 2), dtype=numpy.int32)
    bonds[:9, 0] = numpy.arange(9, dtype=numpy.int32)
    bonds[:9, 1] = numpy.arange(9, dtype=numpy.int32) + 1
    bonds[9:18, 0] = numpy.arange(9, dtype=numpy.int32) + 10
    bonds[9:18, 1] = numpy.arange(9, dtype=numpy.int32) + 11
    bonds[18:27, 0] = numpy.arange(9, dtype=numpy.int32) + 20
    bonds[18:27, 1] = numpy.arange(9, dtype=numpy.int32) + 21
    bonds[27:, 0] = bonds[:27, 1]
    bonds[27:, 1] = bonds[:27:, 0]
    roots = numpy.array([0, 10, 20], dtype=numpy.int32)

    kinforest, kfo_2_to_w_root = kinforest_from_roots_and_bonds(roots, bonds)
    nodes, scans, gens = get_scans(
        kinforest.parent.cpu().numpy(), numpy.zeros((1,), dtype=numpy.int32)
    )

    nodes_to = kfo_2_to_w_root[nodes]

    nodes_gold = numpy.concatenate(
        (
            numpy.arange(11, dtype=numpy.int32),
            numpy.zeros((1,), dtype=numpy.int32),
            numpy.arange(10, dtype=numpy.int32) + 11,
            numpy.zeros((1,), dtype=numpy.int32),
            numpy.arange(10, dtype=numpy.int32) + 21,
        )
    )
    scans_gold = numpy.array([0, 11, 22], dtype=numpy.int32)
    gens_gold = numpy.zeros((2, 2), dtype=numpy.int32)
    gens_gold[1, 0] = 33
    gens_gold[1, 1] = 3

    numpy.testing.assert_equal(nodes_gold, nodes_to)
    numpy.testing.assert_equal(scans_gold, scans)
    numpy.testing.assert_equal(gens_gold, gens)


def test_get_scans_three_simple_branches():
    """Create a forest where each tree has a simple branch
    0->1->2->3(->4->5)->6->7->8->9,
    10->11->12->13(->14->15)->16->17->18->19,
    20->21->22->23(->24->25)->26->27->28->29,
    """
    bonds = numpy.zeros((54, 2), dtype=numpy.int32)
    bonds[:9, 0] = numpy.arange(9, dtype=numpy.int32)
    bonds[:9, 1] = numpy.arange(9, dtype=numpy.int32) + 1
    bonds[5, 0] = 3
    bonds[9:18, 0] = numpy.arange(9, dtype=numpy.int32) + 10
    bonds[9:18, 1] = numpy.arange(9, dtype=numpy.int32) + 11
    bonds[14, 0] = 13
    bonds[18:27, 0] = numpy.arange(9, dtype=numpy.int32) + 20
    bonds[18:27, 1] = numpy.arange(9, dtype=numpy.int32) + 21
    bonds[23, 0] = 23
    bonds[27:, 0] = bonds[:27, 1]
    bonds[27:, 1] = bonds[:27:, 0]

    roots = numpy.array([0, 10, 20], dtype=numpy.int32)

    kinforest, kfo_2_to_w_root = kinforest_from_roots_and_bonds(roots, bonds)
    nodes, scans, gens = get_scans(
        kinforest.parent.cpu().numpy(), numpy.zeros((1,), dtype=numpy.int32)
    )

    nodes_to = kfo_2_to_w_root[nodes]

    nodes_gold = numpy.concatenate(
        (
            numpy.zeros((1,), dtype=numpy.int32),
            numpy.arange(4, dtype=numpy.int32) + 1,
            numpy.arange(4, dtype=numpy.int32) + 7,
            numpy.zeros((1,), dtype=numpy.int32),
            numpy.arange(4, dtype=numpy.int32) + 11,
            numpy.arange(4, dtype=numpy.int32) + 17,
            numpy.zeros((1,), dtype=numpy.int32),
            numpy.arange(4, dtype=numpy.int32) + 21,
            numpy.arange(4, dtype=numpy.int32) + 27,
            numpy.ones((1,), dtype=numpy.int32) * 4,
            numpy.arange(2, dtype=numpy.int32) + 5,
            numpy.ones((1,), dtype=numpy.int32) * 14,
            numpy.arange(2, dtype=numpy.int32) + 15,
            numpy.ones((1,), dtype=numpy.int32) * 24,
            numpy.arange(2, dtype=numpy.int32) + 25,
        )
    )
    scans_gold = numpy.array([0, 9, 18, 0, 3, 6], dtype=numpy.int32)
    gens_gold = numpy.zeros((3, 2), dtype=numpy.int32)
    gens_gold[1, 0] = 27
    gens_gold[1, 1] = 3
    gens_gold[2, 0] = 36
    gens_gold[2, 1] = 6

    numpy.testing.assert_equal(nodes_gold, nodes_to)
    numpy.testing.assert_equal(scans_gold, scans)
    numpy.testing.assert_equal(gens_gold, gens)
