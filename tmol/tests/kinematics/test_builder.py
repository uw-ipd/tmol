import numpy
import torch
import pandas

from tmol.kinematics.operations import inverseKin, forwardKin
from tmol.kinematics.builder import (
    KinematicBuilder,
    stub_defined_for_jump_atom,
    get_c1_and_c2_atoms,
    fix_jump_nodes,
)
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack

# from tmol.score.bonded_atom import BondedAtomScoreGraph
from tmol.score.modules.bonded_atom import BondedAtoms
from tmol.score.modules.bases import ScoreSystem


def test_stub_defined_for_jump_atom_two_descendents_of_jump():
    """Create a forest that is simply a path; jump's stub will be defined
    as it has at least two descendents.
    0->1->2->3->4->5*->6->7->8->9
    """

    child_list_span = numpy.zeros((10, 2), dtype=numpy.int32)
    for i in range(9):
        child_list_span[i, 0] = i
        child_list_span[i, 1] = i + 1
    child_list_span[9, :] = 10

    child_list = numpy.arange(9, dtype=numpy.int32) + 1

    atom_is_jump = numpy.zeros((10,), dtype=bool)
    atom_is_jump[5] = True

    is_defined = stub_defined_for_jump_atom(
        5, atom_is_jump, child_list_span, child_list
    )
    assert is_defined


def test_stub_defined_for_jump_atom_two_direct_children_of_jump():
    """
    0->1->2->3->4->5*(->6)->7->8->9
    """

    child_list_span = numpy.zeros((10, 2), dtype=numpy.int32)
    jump_atom = 5
    for i in range(9):
        if i < jump_atom or i > jump_atom + 1:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 1
        if i == jump_atom:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 2
        if i == jump_atom + 1:
            child_list_span[i, 0] = i + 1
            child_list_span[i, 1] = i + 1

    child_list_span[9, :] = 10

    child_list = numpy.arange(9, dtype=numpy.int32) + 1

    atom_is_jump = numpy.zeros((10,), dtype=bool)
    atom_is_jump[5] = True

    is_defined = stub_defined_for_jump_atom(
        5, atom_is_jump, child_list_span, child_list
    )
    assert is_defined


def test_stub_defined_for_jump_atom_w_jump_children_of_jump():
    """
    0->1->2->3->4->5*(->6*)->7->8->9
    """

    child_list_span = numpy.zeros((10, 2), dtype=numpy.int32)
    jump_atom = 5
    for i in range(9):
        if i < jump_atom or i > jump_atom + 1:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 1
        if i == jump_atom:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 2
        if i == jump_atom + 1:
            child_list_span[i, 0] = i + 1
            child_list_span[i, 1] = i + 1

    child_list_span[9, :] = 10

    child_list = numpy.arange(9, dtype=numpy.int32) + 1

    atom_is_jump = numpy.zeros((10,), dtype=bool)
    atom_is_jump[jump_atom] = True
    atom_is_jump[jump_atom + 1] = True

    is_defined = stub_defined_for_jump_atom(
        5, atom_is_jump, child_list_span, child_list
    )
    assert is_defined


def test_stub_defined_for_jump_atom_w_insufficient_children_excluding_jumps():
    """
    0->1->2->3->4->5*(->6*)->7
    """

    child_list_span = numpy.zeros((8, 2), dtype=numpy.int32)
    jump_atom = 5
    for i in range(7):
        if i < jump_atom or i > jump_atom + 1:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 1
        if i == jump_atom:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 2
        if i == jump_atom + 1:
            child_list_span[i, 0] = i + 1
            child_list_span[i, 1] = i + 1

    child_list_span[6, :] = 7

    child_list = numpy.arange(7, dtype=numpy.int32) + 1

    atom_is_jump = numpy.zeros((8,), dtype=bool)
    atom_is_jump[jump_atom] = True
    atom_is_jump[jump_atom + 1] = True

    is_defined = stub_defined_for_jump_atom(
        5, atom_is_jump, child_list_span, child_list
    )
    assert not is_defined


def test_get_c1_and_c2_atoms_for_jump_atom_two_descendents_of_jump():
    """Create a forest that is simply a path; jump's stub will be defined
    as it has at least two descendents.
    0->1->2->3->4->5*->6->7->8->9
    """

    child_list_span = numpy.zeros((10, 2), dtype=numpy.int32)
    for i in range(9):
        child_list_span[i, 0] = i
        child_list_span[i, 1] = i + 1
    child_list_span[9, :] = 10

    child_list = numpy.arange(9, dtype=numpy.int32) + 1
    parents = numpy.arange(10, dtype=numpy.int32) - 1
    parents[0] = 0

    atom_is_jump = numpy.zeros((10,), dtype=bool)
    atom_is_jump[5] = True

    c1, c2 = get_c1_and_c2_atoms(5, atom_is_jump, child_list_span, child_list, parents)
    assert c1 == 6
    assert c2 == 7


def test_get_c1_and_c2_atoms_for_jump_atom_two_direct_children_of_jump():
    """
    0->1->2->3->4->5*(->6)->7->8->9
    """

    child_list_span = numpy.zeros((10, 2), dtype=numpy.int32)
    jump_atom = 5
    for i in range(9):
        if i < jump_atom or i > jump_atom + 1:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 1
        if i == jump_atom:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 2
        if i == jump_atom + 1:
            child_list_span[i, 0] = i + 1
            child_list_span[i, 1] = i + 1

    child_list_span[9, :] = 10

    child_list = numpy.arange(9, dtype=numpy.int32) + 1

    parents = numpy.arange(10, dtype=numpy.int32) - 1
    parents[0] = 0
    parents[jump_atom + 2] = jump_atom

    atom_is_jump = numpy.zeros((10,), dtype=bool)
    atom_is_jump[5] = True

    c1, c2 = get_c1_and_c2_atoms(5, atom_is_jump, child_list_span, child_list, parents)
    assert c1 == 6
    assert c2 == 7


def test_get_c1_and_c2_atoms_for_jump_atom_w_jump_child():
    """
    0->1->2->3->4->5*(->6*)->7->8->9
    """

    child_list_span = numpy.zeros((10, 2), dtype=numpy.int32)
    jump_atom = 5
    for i in range(9):
        if i < jump_atom or i > jump_atom + 1:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 1
        if i == jump_atom:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 2
        if i == jump_atom + 1:
            child_list_span[i, 0] = i + 1
            child_list_span[i, 1] = i + 1

    child_list_span[9, :] = 10

    child_list = numpy.arange(9, dtype=numpy.int32) + 1
    parents = numpy.arange(10, dtype=numpy.int32) - 1
    parents[0] = 0
    parents[jump_atom + 2] = jump_atom

    atom_is_jump = numpy.zeros((10,), dtype=bool)
    atom_is_jump[jump_atom] = True
    atom_is_jump[jump_atom + 1] = True

    c1, c2 = get_c1_and_c2_atoms(5, atom_is_jump, child_list_span, child_list, parents)
    assert c1 == 7
    assert c2 == 8


def test_get_c1_and_c2_atoms_for_jump_atom_w_insufficient_children_excluding_jumps():
    """
    0->1->2->3->4->5*(->6*)->7
    """

    child_list_span = numpy.zeros((8, 2), dtype=numpy.int32)
    jump_atom = 5
    for i in range(7):
        if i < jump_atom or i > jump_atom + 1:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 1
        if i == jump_atom:
            child_list_span[i, 0] = i
            child_list_span[i, 1] = i + 2
        if i == jump_atom + 1:
            child_list_span[i, 0] = i + 1
            child_list_span[i, 1] = i + 1

    child_list_span[6, :] = 7

    child_list = numpy.arange(7, dtype=numpy.int32) + 1
    parents = numpy.arange(8, dtype=numpy.int32) - 1
    parents[0] = 0
    parents[jump_atom + 2] = jump_atom

    atom_is_jump = numpy.zeros((8,), dtype=bool)
    atom_is_jump[jump_atom] = True
    atom_is_jump[jump_atom + 1] = True

    c1, c2 = get_c1_and_c2_atoms(5, atom_is_jump, child_list_span, child_list, parents)

    # we will have recursed way up the forest until we find a parent
    # with two descendants within two generations
    assert c1 == 3
    assert c2 == 4


def test_fix_jump_nodes_one_root_node_path():
    """*0->1->2->3->4->5->6->7->8->9"""
    parents = numpy.arange(10, dtype=numpy.int32) - 1
    parents[0] = 0

    frame_x = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z = numpy.full((10,), -1, dtype=numpy.int32)

    roots = numpy.zeros((1,), dtype=numpy.int32)
    jumps = numpy.array([], dtype=numpy.int32)

    fix_jump_nodes(parents, frame_x, frame_y, frame_z, roots, jumps)

    assert frame_x[0] == 1
    assert frame_y[0] == 0
    assert frame_z[0] == 2

    assert frame_x[1] == 1
    assert frame_y[1] == 0
    assert frame_z[1] == 2

    assert numpy.all(frame_x[2:] == -1)
    assert numpy.all(frame_y[2:] == -1)
    assert numpy.all(frame_z[2:] == -1)


def test_fix_jump_nodes__one_jump():
    """0->1->2->3->4->5*->6->7->8->9"""
    parents = numpy.arange(10, dtype=numpy.int32) - 1
    parents[0] = 0
    frame_x = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z = numpy.full((10,), -1, dtype=numpy.int32)
    roots = numpy.array([], dtype=numpy.int32)
    jumps = numpy.full((1), 5, dtype=numpy.int32)

    fix_jump_nodes(parents, frame_x, frame_y, frame_z, roots, jumps)

    frame_x_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z_gold = numpy.full((10,), -1, dtype=numpy.int32)

    frame_x_gold[[5, 6]] = 6
    frame_y_gold[[5, 6]] = 5
    frame_z_gold[[5, 6]] = 7

    numpy.testing.assert_equal(frame_x_gold, frame_x)
    numpy.testing.assert_equal(frame_y_gold, frame_y)
    numpy.testing.assert_equal(frame_z_gold, frame_z)


def test_fix_jump_nodes__one_root_one_jump():
    """*0->1->2->3->4->5*->6->7->8->9"""
    parents = numpy.arange(10, dtype=numpy.int32) - 1
    parents[0] = 0
    frame_x = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z = numpy.full((10,), -1, dtype=numpy.int32)
    roots = numpy.full((1), 0, dtype=numpy.int32)
    jumps = numpy.full((1), 5, dtype=numpy.int32)

    fix_jump_nodes(parents, frame_x, frame_y, frame_z, roots, jumps)

    frame_x_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z_gold = numpy.full((10,), -1, dtype=numpy.int32)

    frame_x_gold[[0, 1]] = 1
    frame_y_gold[[0, 1]] = 0
    frame_z_gold[[0, 1]] = 2

    frame_x_gold[[5, 6]] = 6
    frame_y_gold[[5, 6]] = 5
    frame_z_gold[[5, 6]] = 7

    numpy.testing.assert_equal(frame_x_gold, frame_x)
    numpy.testing.assert_equal(frame_y_gold, frame_y)
    numpy.testing.assert_equal(frame_z_gold, frame_z)


def test_fix_jump_nodes__one_root_one_jump_first_descendent_w_no_children():
    """*0(->1)->2->3->4->5*(->6)->7->8->9"""
    parents = numpy.arange(10, dtype=numpy.int32) - 1
    parents[0] = 0
    parents[2] = 0
    parents[7] = 5
    frame_x = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z = numpy.full((10,), -1, dtype=numpy.int32)
    roots = numpy.full((1), 0, dtype=numpy.int32)
    jumps = numpy.full((1), 5, dtype=numpy.int32)

    fix_jump_nodes(parents, frame_x, frame_y, frame_z, roots, jumps)

    frame_x_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z_gold = numpy.full((10,), -1, dtype=numpy.int32)

    frame_x_gold[[0, 1]] = 1
    frame_y_gold[[0, 1]] = 0
    frame_z_gold[[0, 1]] = 2

    frame_x_gold[2] = 2
    frame_y_gold[2] = 0
    frame_z_gold[2] = 1

    frame_x_gold[[5, 6]] = 6
    frame_y_gold[[5, 6]] = 5
    frame_z_gold[[5, 6]] = 7

    frame_x_gold[7] = 7
    frame_y_gold[7] = 5
    frame_z_gold[7] = 6

    numpy.testing.assert_equal(frame_x_gold, frame_x)
    numpy.testing.assert_equal(frame_y_gold, frame_y)
    numpy.testing.assert_equal(frame_z_gold, frame_z)


def test_fix_jump_nodes__one_root_one_jump_first_descendent_w_no_children2():
    """*0(->*1)(->2*)->3->4->5*(->6)->7->8->9"""
    parents = numpy.arange(10, dtype=numpy.int32) - 1
    parents[0] = 0
    parents[2] = 0
    parents[3] = 0
    parents[7] = 5
    frame_x = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z = numpy.full((10,), -1, dtype=numpy.int32)
    roots = numpy.full((1), 0, dtype=numpy.int32)
    jumps = numpy.array([1, 2, 5], dtype=numpy.int32)

    fix_jump_nodes(parents, frame_x, frame_y, frame_z, roots, jumps)

    frame_x_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z_gold = numpy.full((10,), -1, dtype=numpy.int32)

    frame_x_gold[[0, 1, 2, 3]] = 3
    frame_y_gold[[0, 3]] = 0
    frame_z_gold[[0, 1, 2, 3]] = 4

    frame_y_gold[1] = 1
    frame_y_gold[2] = 2

    frame_x_gold[[5, 6]] = 6
    frame_y_gold[[5, 6]] = 5
    frame_z_gold[[5, 6]] = 7

    frame_x_gold[7] = 7
    frame_y_gold[7] = 5
    frame_z_gold[7] = 6

    numpy.testing.assert_equal(frame_x_gold, frame_x)
    numpy.testing.assert_equal(frame_y_gold, frame_y)
    numpy.testing.assert_equal(frame_z_gold, frame_z)


def test_fix_jump_nodes__jump_w_one_jump_child_and_one_nonjump_child():
    """*0(->*1(->2*))->3)->4->5(->6)->7->8->9
    Note that I am uncertain whether this test is accurately capturing
    the desired behavior.  I am merely testing that the code is working
    "as written." There is still an outstanding  "TO DO" when defining
    the coordinate frame for  the non-jump children of a jump for which
    stub_defined_for_jump_atom return False
    """
    parents = numpy.arange(10, dtype=numpy.int32) - 1
    parents[0] = 0
    parents[3] = 1
    parents[4] = 0
    parents[7] = 5
    frame_x = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z = numpy.full((10,), -1, dtype=numpy.int32)
    roots = numpy.full((1), 0, dtype=numpy.int32)
    jumps = numpy.array([1, 2], dtype=numpy.int32)

    fix_jump_nodes(parents, frame_x, frame_y, frame_z, roots, jumps)

    frame_x_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z_gold = numpy.full((10,), -1, dtype=numpy.int32)

    frame_x_gold[[0, 1, 2, 3, 4]] = 4
    frame_y_gold[[0, 4]] = 0
    frame_z_gold[[0, 1, 2, 3, 4]] = 5

    frame_y_gold[[1, 3]] = 1
    frame_y_gold[2] = 2

    numpy.testing.assert_equal(frame_x_gold, frame_x)
    numpy.testing.assert_equal(frame_y_gold, frame_y)
    numpy.testing.assert_equal(frame_z_gold, frame_z)


def test_fix_jump_nodes__one_root_one_jump_many_children_of_both():
    """*0(->1->2)(->3)(->4)->5*(->6->7)(->8)(->9)"""
    parents = numpy.arange(10, dtype=numpy.int32) - 1
    parents[[0, 3, 4, 5]] = 0
    parents[[8, 9]] = 5
    frame_x = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z = numpy.full((10,), -1, dtype=numpy.int32)
    roots = numpy.full((1), 0, dtype=numpy.int32)
    jumps = numpy.full((1), 5, dtype=numpy.int32)

    fix_jump_nodes(parents, frame_x, frame_y, frame_z, roots, jumps)

    frame_x_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_y_gold = numpy.full((10,), -1, dtype=numpy.int32)
    frame_z_gold = numpy.full((10,), -1, dtype=numpy.int32)

    frame_x_gold[[0, 1]] = 1
    frame_y_gold[[0, 1]] = 0
    frame_z_gold[[0, 1]] = 2

    frame_x_gold[[3, 4]] = numpy.arange(2, dtype=numpy.int32) + 3
    frame_y_gold[[3, 4]] = 0
    frame_z_gold[[3, 4]] = 1

    frame_x_gold[[5, 6]] = 6
    frame_y_gold[[5, 6]] = 5
    frame_z_gold[[5, 6]] = 7

    frame_x_gold[[8, 9]] = numpy.arange(2, dtype=numpy.int32) + 8
    frame_y_gold[[8, 9]] = 5
    frame_z_gold[[8, 9]] = 6

    numpy.testing.assert_equal(frame_x_gold, frame_x)
    numpy.testing.assert_equal(frame_y_gold, frame_y)
    numpy.testing.assert_equal(frame_z_gold, frame_z)


def test_fix_jump_nodes__one_root_one_jump_w_jump_descendent():
    """*0(->1)->2->3->4->5*(->6*->7->8)->9->10"""
    parents = numpy.arange(11, dtype=numpy.int32) - 1
    parents[0] = 0
    parents[2] = 0
    parents[9] = 5

    frame_x = numpy.full((11,), -1, dtype=numpy.int32)
    frame_y = numpy.full((11,), -1, dtype=numpy.int32)
    frame_z = numpy.full((11,), -1, dtype=numpy.int32)
    roots = numpy.full((1), 0, dtype=numpy.int32)
    jumps = numpy.array([5, 6], dtype=numpy.int32)

    fix_jump_nodes(parents, frame_x, frame_y, frame_z, roots, jumps)

    frame_x_gold = numpy.full((11,), -1, dtype=numpy.int32)
    frame_y_gold = numpy.full((11,), -1, dtype=numpy.int32)
    frame_z_gold = numpy.full((11,), -1, dtype=numpy.int32)

    frame_x_gold[[0, 1]] = 1
    frame_y_gold[[0, 1]] = 0
    frame_z_gold[[0, 1]] = 2

    frame_x_gold[2] = 2
    frame_y_gold[2] = 0
    frame_z_gold[2] = 1

    frame_x_gold[[5, 9]] = 9
    frame_y_gold[[5, 9]] = 5
    frame_z_gold[[5, 9]] = 10

    frame_x_gold[[6, 7]] = 7
    frame_y_gold[[6, 7]] = 6
    frame_z_gold[[6, 7]] = 8

    numpy.testing.assert_equal(frame_x_gold, frame_x)
    numpy.testing.assert_equal(frame_y_gold, frame_y)
    numpy.testing.assert_equal(frame_z_gold, frame_z)


def test_fix_jump_nodes__one_root_one_jump_w_jump_descendents():
    """*0(->1)->2->3->4->5*(->6*->7->8)->(9*->10->11)->12->13"""
    parents = numpy.arange(14, dtype=numpy.int32) - 1
    parents[0] = 0
    parents[2] = 0
    parents[9] = 5
    parents[12] = 5

    frame_x = numpy.full((14,), -1, dtype=numpy.int32)
    frame_y = numpy.full((14,), -1, dtype=numpy.int32)
    frame_z = numpy.full((14,), -1, dtype=numpy.int32)
    roots = numpy.full((1), 0, dtype=numpy.int32)
    jumps = numpy.array([5, 6, 9], dtype=numpy.int32)

    fix_jump_nodes(parents, frame_x, frame_y, frame_z, roots, jumps)

    frame_x_gold = numpy.full((14,), -1, dtype=numpy.int32)
    frame_y_gold = numpy.full((14,), -1, dtype=numpy.int32)
    frame_z_gold = numpy.full((14,), -1, dtype=numpy.int32)

    frame_x_gold[[0, 1]] = 1
    frame_y_gold[[0, 1]] = 0
    frame_z_gold[[0, 1]] = 2

    frame_x_gold[2] = 2
    frame_y_gold[2] = 0
    frame_z_gold[2] = 1

    frame_x_gold[[5, 12]] = 12
    frame_y_gold[[5, 12]] = 5
    frame_z_gold[[5, 12]] = 13

    frame_x_gold[[6, 7]] = 7
    frame_y_gold[[6, 7]] = 6
    frame_z_gold[[6, 7]] = 8

    frame_x_gold[[9, 10]] = 10
    frame_y_gold[[9, 10]] = 9
    frame_z_gold[[9, 10]] = 11

    numpy.testing.assert_equal(frame_x_gold, frame_x)
    numpy.testing.assert_equal(frame_y_gold, frame_y)
    numpy.testing.assert_equal(frame_z_gold, frame_z)


def test_build_er_bonds_to_csgraph():
    """
    *0->1->2->3->4->5->6->7->8->9
    """

    bonds = numpy.zeros((18, 2), dtype=numpy.int32)
    bonds[:9, 0] = numpy.arange(9, dtype=numpy.int32)
    bonds[:9, 1] = numpy.arange(9, dtype=numpy.int32) + 1
    bonds[9:, 0] = bonds[:9, 1]
    bonds[9:, 1] = bonds[:9:, 0]

    csr_mat = KinematicBuilder.bonds_to_csgraph(9, bonds)
    mat = csr_mat.toarray()

    mat_gold = numpy.zeros((10, 10), dtype=numpy.float32)
    mat_gold[bonds[:, 0], bonds[:, 1]] = 1

    numpy.testing.assert_equal(mat_gold, mat)


def test_builder_define_forest_with_prioritized_bonds():
    """
    *0->1->2->3->4->5->6->7->8->9
    """
    potential_bonds = numpy.zeros((18, 2), dtype=numpy.int32)
    prioritized_bonds = numpy.zeros([0, 2], dtype=numpy.int32)

    potential_bonds[:9, 0] = numpy.arange(9, dtype=numpy.int32)
    potential_bonds[:9, 1] = numpy.arange(9, dtype=numpy.int32) + 1
    potential_bonds[9:, 0] = potential_bonds[:9, 1]
    potential_bonds[9:, 1] = potential_bonds[:9:, 0]

    roots = numpy.full((1,), 0, dtype=numpy.int32)

    (
        kfo_2_to,
        to_parents_in_kfo,
    ) = KinematicBuilder().define_trees_with_prioritized_bonds(
        roots, potential_bonds, prioritized_bonds, 10
    )

    kfo_2_to_gold = numpy.arange(10, dtype=numpy.int32)
    to_parents_in_kfo_gold = numpy.arange(10, dtype=numpy.int32) - 1
    to_parents_in_kfo_gold[0] = -9999

    numpy.testing.assert_equal(kfo_2_to_gold, kfo_2_to)
    numpy.testing.assert_equal(to_parents_in_kfo_gold, to_parents_in_kfo)


def test_builder_define_forest_with_prioritized_bonds2():
    """
    Test single integer input for "roots" variable:
    *0->1->2->3->4->5->6->7->8->9
    """
    potential_bonds = numpy.zeros((18, 2), dtype=numpy.int32)
    prioritized_bonds = numpy.zeros([0, 2], dtype=numpy.int32)

    potential_bonds[:9, 0] = numpy.arange(9, dtype=numpy.int32)
    potential_bonds[:9, 1] = numpy.arange(9, dtype=numpy.int32) + 1
    potential_bonds[9:, 0] = potential_bonds[:9, 1]
    potential_bonds[9:, 1] = potential_bonds[:9:, 0]

    roots = 0

    (
        kfo_2_to,
        to_parents_in_kfo,
    ) = KinematicBuilder().define_trees_with_prioritized_bonds(
        roots, potential_bonds, prioritized_bonds, 10
    )

    kfo_2_to_gold = numpy.arange(10, dtype=numpy.int32)
    to_parents_in_kfo_gold = numpy.arange(10, dtype=numpy.int32) - 1
    to_parents_in_kfo_gold[0] = -9999

    numpy.testing.assert_equal(kfo_2_to_gold, kfo_2_to)
    numpy.testing.assert_equal(to_parents_in_kfo_gold, to_parents_in_kfo)


def test_builder_refold(ubq_system):
    tsys = ubq_system

    bonds32 = numpy.array(tsys.bonds, dtype=numpy.int32)
    id, parents = KinematicBuilder.bonds_to_forest(
        numpy.array([0], dtype=numpy.int32), bonds32
    )
    kinforest = (
        KinematicBuilder()
        .append_connected_components(
            to_roots=numpy.array([0], dtype=numpy.int32),
            kfo_2_to=id,
            to_parents_in_kfo=parents,
            to_jump_nodes=numpy.array([], dtype=numpy.int32),
        )
        .kinforest
    )

    kincoords = torch.DoubleTensor(tsys.coords[kinforest.id])
    dofs = inverseKin(kinforest, kincoords)
    refold_kincoords = forwardKin(kinforest, dofs)

    assert (refold_kincoords[0] == 0).all()

    refold_coords = numpy.full_like(tsys.coords, numpy.nan)
    refold_coords[kinforest.id[1:].squeeze()] = refold_kincoords[1:]

    numpy.testing.assert_allclose(tsys.coords, refold_coords)


def test_builder_framing(ubq_system):
    """Test first-three-atom framing logic in kinematic builder."""

    tsys = ubq_system
    roots = numpy.array([0], dtype=numpy.int32)
    ids, parents = KinematicBuilder.bonds_to_forest(
        roots=roots, bonds=tsys.bonds.astype(numpy.int32)
    )
    kinforest = (
        KinematicBuilder()
        .append_connected_components(
            to_roots=roots, kfo_2_to=ids, to_parents_in_kfo=parents, to_jump_nodes=[]
        )
        .kinforest
    )

    # The first entries in the forest should be the global DOF root, self-parented,
    # followed by the first atom.
    root_children = kinforest[kinforest.parent == 0]
    assert len(root_children) == 2
    numpy.testing.assert_array_equal(kinforest.parent[:2], [0, 0])
    numpy.testing.assert_array_equal(kinforest.id[:2], [-1, 0])

    atom_root_children = numpy.flatnonzero(numpy.array(kinforest.parent) == 1)
    atom_root_grandkids = numpy.flatnonzero(
        numpy.array(kinforest.parent) == atom_root_children[0]
    )
    assert len(atom_root_children) == 2
    assert len(atom_root_grandkids) == 3

    # The first atom has two children. The first atom and its first child are framed by
    # [first_child, root, first_grandkid]
    first_atom = kinforest[1]
    assert int(first_atom.frame_x) == atom_root_children[0]
    assert int(first_atom.frame_y) == 1
    assert int(first_atom.frame_z) == atom_root_grandkids[0]

    first_atom_first_child = kinforest[atom_root_children[0]]
    assert int(first_atom_first_child.frame_x) == atom_root_children[0]
    assert int(first_atom_first_child.frame_y) == 1
    assert int(first_atom_first_child.frame_z) == atom_root_grandkids[0]

    # The rest of the children are framed by:
    # [self, root, first_child]
    for c in atom_root_children[1:]:
        first_atom_other_child = kinforest[c]
        assert int(first_atom_other_child.frame_x) == c
        assert int(first_atom_other_child.frame_y) == 1
        assert int(first_atom_other_child.frame_z) == atom_root_children[0]

    # Other atoms are framed normally, [self, parent, grandparent]
    normal_atoms = numpy.flatnonzero(numpy.array(kinforest.parent > 1))
    numpy.testing.assert_array_equal(kinforest.frame_x[normal_atoms], normal_atoms)
    numpy.testing.assert_array_equal(
        kinforest.frame_y[normal_atoms], kinforest.parent[normal_atoms]
    )
    numpy.testing.assert_array_equal(
        kinforest.frame_z[normal_atoms],
        kinforest.parent[kinforest.parent[normal_atoms].to(dtype=torch.long)],
    )


def test_build_two_system_kinematics(ubq_system, torch_device):
    natoms = numpy.sum(numpy.logical_not(numpy.isnan(ubq_system.coords[:, 0])))

    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    # bonds = BondedAtomScoreGraph.build_for(twoubq, device=torch_device).bonds.astype(
    #    numpy.int32
    # )
    system = ScoreSystem._build_with_modules(twoubq, {BondedAtoms})
    bonds = BondedAtoms.get(system).bonds.astype(numpy.int32)

    # print("bonds", bonds.shape)
    tworoots = numpy.array((0, twoubq.systems[0].system_size), dtype=numpy.int32)
    syssize = twoubq.systems[0].system_size

    bonds_compact = bonds[:, 0:1] * syssize + bonds[:, 1:3]
    # print("bonds_compact", bonds_compact.shape)
    # print("bonds_compact", bonds_compact[:10])

    ids, parents = KinematicBuilder.bonds_to_forest(roots=tworoots, bonds=bonds_compact)
    # print("ids")
    # print(ids[:20])
    # print("parents")
    # print(parents[:20])

    # self_inds = numpy.arange(parents.shape[0], dtype=int)
    # print("self parents")
    # print(self_inds[parents[self_inds] == self_inds])

    id_index = pandas.Index(ids)
    root_index = id_index.get_indexer(tworoots)

    builder = KinematicBuilder()
    forest = builder.append_connected_components(
        to_roots=tworoots, kfo_2_to=ids, to_parents_in_kfo=parents, to_jump_nodes=[]
    ).kinforest

    assert forest.id.shape[0] == 2 * natoms + 1
    assert forest.id[1] == 0
    assert forest.parent[1 + root_index[0]] == 0
    assert forest.parent[1 + root_index[1]] == 0


def test_build_jagged_system(ubq_res, torch_device):
    ubq40 = PackedResidueSystem.from_residues(ubq_res[:1])
    ubq60 = PackedResidueSystem.from_residues(ubq_res[:2])
    natoms = numpy.sum(numpy.logical_not(numpy.isnan(ubq40.coords[:, 0]))) + numpy.sum(
        numpy.logical_not(numpy.isnan(ubq60.coords[:, 0]))
    )
    twoubq = PackedResidueSystemStack((ubq40, ubq60))
    # bonds = BondedAtomScoreGraph.build_for(twoubq, device=torch_device).bonds.astype(
    #    numpy.int32
    # )
    system = ScoreSystem._build_with_modules(twoubq, {BondedAtoms})
    bonds = BondedAtoms.get(system).bonds.astype(numpy.int32)

    tworoots = numpy.array((0, twoubq.systems[1].system_size), dtype=numpy.int32)
    syssize = twoubq.systems[1].system_size
    bonds_compact = bonds[:, 0:1] * syssize + bonds[:, 1:3]

    ids, parents = KinematicBuilder.bonds_to_forest(roots=tworoots, bonds=bonds_compact)

    # print("ids")
    # print(ids)
    # print("parents")
    # print(parents)

    id_index = pandas.Index(ids)
    root_index = id_index.get_indexer(tworoots)
    # print("root_index")
    # print(root_index)

    builder = KinematicBuilder()
    forest = builder.append_connected_components(
        to_roots=tworoots, kfo_2_to=ids, to_parents_in_kfo=parents, to_jump_nodes=[]
    ).kinforest

    assert forest.id.shape[0] == natoms + 1
    assert forest.id[1] == 0
    assert forest.parent[1 + root_index[0]] == 0
    assert forest.parent[1 + root_index[1]] == 0

    # print("forest.parent[1 + root_index[0]]")
    # print(forest.parent[1 + root_index[0]])
    # print("forest.parent[1 + root_index[1]]")
    # print(forest.parent[1 + root_index[1]])
