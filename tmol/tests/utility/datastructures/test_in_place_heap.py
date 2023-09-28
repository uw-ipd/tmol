import pytest
import torch
import numpy

# import math

import tmol.utility.cpp_extension as cpp_extension
from tmol.utility.cpp_extension import relpaths, modulename

from tmol.tests.torch import requires_cuda


@pytest.fixture
def in_place_heap():
    return cpp_extension.load(
        modulename(__name__), relpaths(__file__, "in_place_heap.cpp"), verbose=True
    )


@pytest.fixture
def reverse_insert10_heap_structure():
    # imagine inserting torch.flip(arange(10)) into a heap: what happens?
    #
    # logically, how will this sequence of insertions work?
    # [(9, 0)] insert node 0, val 9
    # [(8,1) (9,0)] insert node 1, val 8
    # [(7,2) (9,0) (8,1)] insert node 2, val 7
    # [(6,3) (7,2) (8,1) (9,0)] insert node 3, val 6; bubble 0, bubble 2
    # [(5,4) (6,3) (8,1) (9,0) (7,2)] i4,v5: b2, b3
    # [(5,4) (6,3) (8,1) (9,0) (7,2) (4,5)] i5,v4: step0
    # [(5,4) (6,3) (4,5) (9,0) (7,2) (8,1)] i5,v4: step1 bubble 1
    # [(4,5) (6,3) (5,4) (9,0) (7,2) (8,1)] i5,v4: step2 bubble 4

    # [(4,5) (6,3) (5,4) (9,0) (7,2) (8,1) (3,6)] i6,v3: step0
    # [(4,5) (6,3) (3,6) (9,0) (7,2) (8,1) (5,4)] i6,v3: step1: bubble 4
    # [(3,6) (6,3) (4,5) (9,0) (7,2) (8,1) (5,4)] i6,v3: step2: bubble 5

    # [(3,6) (6,3) (4,5) (9,0) (7,2) (8,1) (5,4) (2,7)] i7,v2: step0
    # [(3,6) (6,3) (4,5) (2,7) (7,2) (8,1) (5,4) (9,0)] i7,v2: step1: bubble 0
    # [(3,6) (2,7) (4,5) (6,3) (7,2) (8,1) (5,4) (9,0)] i7,v2: step2: bubble 3
    # [(2,7) (3,6) (4,5) (6,3) (7,2) (8,1) (5,4) (9,0)] i7,v2: step3: bubble 6

    # [(2,7) (3,6) (4,5) (6,3) (7,2) (8,1) (5,4) (9,0) (1,8)] i8,v1: step0
    # [(2,7) (3,6) (4,5) (1,8) (7,2) (8,1) (5,4) (9,0) (6,3)] i8,v1: step1: bubble 3
    # [(2,7) (1,8) (4,5) (3,6) (7,2) (8,1) (5,4) (9,0) (6,3)] i8,v1: step2: bubble 6
    # [(1,8) (2,7) (4,5) (3,6) (7,2) (8,1) (5,4) (9,0) (6,3)] i8,v1: step2: bubble 7

    # [(1,8) (2,7) (4,5) (3,6) (7,2) (8,1) (5,4) (9,0) (6,3) (0,9)] i9,v0: step0
    # [(1,8) (2,7) (4,5) (3,6) (0,9) (8,1) (5,4) (9,0) (6,3) (7,2)] i9,v0: step1 bubble 2
    # [(1,8) (0,9) (4,5) (3,6) (2,7) (8,1) (5,4) (9,0) (6,3) (7,2)] i9,v0: step1 bubble 7
    # [(0,9) (1,8) (4,5) (3,6) (2,7) (8,1) (5,4) (9,0) (6,3) (7,2)] i9,v0: step1 bubble 8

    gold_heap_order = numpy.array([9, 8, 5, 6, 7, 1, 4, 0, 3, 2], dtype=numpy.int32)
    gold_node_order = numpy.array([7, 5, 9, 8, 6, 2, 3, 4, 1, 0], dtype=numpy.int32)
    gold_values = numpy.array([0, 1, 4, 3, 2, 8, 5, 9, 6, 7], dtype=numpy.int32)

    return gold_heap_order, gold_node_order, gold_values


def test_heap_construction_1(in_place_heap):
    # If the nodes are inserted in increasing value, their
    # order should be preserved
    tvec = torch.arange(10, dtype=torch.int32)
    node_order, heap_order, values = in_place_heap.create_in_place_heap(tvec)
    numpy.testing.assert_equal(tvec.numpy(), node_order.numpy())
    numpy.testing.assert_equal(tvec.numpy(), heap_order.numpy())
    numpy.testing.assert_equal(tvec.numpy(), values.numpy())


def test_heap_construction_2(in_place_heap, reverse_insert10_heap_structure):
    tvec = torch.flip(torch.arange(10, dtype=torch.int32), dims=[0])
    node_order, heap_order, values = in_place_heap.create_in_place_heap(tvec)
    gold_heap_order, gold_node_order, gold_values = reverse_insert10_heap_structure

    numpy.testing.assert_equal(gold_node_order, node_order.numpy())
    numpy.testing.assert_equal(gold_heap_order, heap_order.numpy())
    numpy.testing.assert_equal(gold_values, values.numpy())


def test_heap_clear_and_reconstruction(in_place_heap, reverse_insert10_heap_structure):
    tvec1 = torch.arange(10, dtype=torch.int32)
    tvec2 = torch.flip(torch.arange(10, dtype=torch.int32), dims=[0])
    node_order, heap_order, values = in_place_heap.clear_heap_after_creation(
        tvec1, tvec2
    )
    gold_heap_order, gold_node_order, gold_values = reverse_insert10_heap_structure

    numpy.testing.assert_equal(gold_node_order, node_order.numpy())
    numpy.testing.assert_equal(gold_heap_order, heap_order.numpy())
    numpy.testing.assert_equal(gold_values, values.numpy())


def test_heap_clear_and_reconstruction_smaller_subset(
    in_place_heap, reverse_insert10_heap_structure
):
    tvec1 = torch.arange(20, dtype=torch.int32)
    tvec2 = torch.flip(torch.arange(10, dtype=torch.int32), dims=[0])
    node_order, heap_order, values = in_place_heap.clear_heap_after_creation(
        tvec1, tvec2
    )
    gold_heap_order, gold_node_order, gold_values = reverse_insert10_heap_structure

    numpy.testing.assert_equal(gold_node_order, node_order.numpy())
    numpy.testing.assert_equal(gold_heap_order, heap_order.numpy())
    numpy.testing.assert_equal(gold_values, values.numpy())


def test_heap_with_gaps(in_place_heap, reverse_insert10_heap_structure):
    node_ind = 2 * torch.arange(10, dtype=torch.int32)
    values = torch.flip(torch.arange(10, dtype=torch.int32), dims=[0])

    seq = torch.cat([node_ind.unsqueeze(1), values.unsqueeze(1)], dim=1)
    node_order, heap_order, values = in_place_heap.create_heap_with_gaps(20, seq)
    gold_heap_order, gold_node_order, gold_values = reverse_insert10_heap_structure

    gold_node_order2 = numpy.full((20,), -1, dtype=numpy.int32)
    gold_heap_order2 = numpy.full((20,), -1, dtype=numpy.int32)
    gold_values2 = numpy.full((20,), -1, dtype=numpy.int32)
    gold_node_order2[node_ind] = gold_node_order
    gold_heap_order2[:10] = 2 * gold_heap_order
    gold_values2[:10] = gold_values

    numpy.testing.assert_equal(gold_node_order2, node_order.numpy())
    numpy.testing.assert_equal(gold_heap_order2, heap_order.numpy())
    numpy.testing.assert_equal(gold_values2, values.numpy())


def test_heap_with_gaps2(in_place_heap, reverse_insert10_heap_structure):
    node_ind1 = 2 * torch.arange(10, dtype=torch.int32)
    node_ind2 = 2 * torch.arange(10, dtype=torch.int32) + 1
    values = torch.flip(torch.arange(10, dtype=torch.int32), dims=[0])

    seq1 = torch.cat([node_ind1.unsqueeze(1), values.unsqueeze(1)], dim=1)
    seq2 = torch.cat([node_ind2.unsqueeze(1), values.unsqueeze(1)], dim=1)
    (
        node_order,
        heap_order,
        values,
    ) = in_place_heap.create_heap_with_gaps_clear_and_recreate(20, seq1, seq2)
    gold_heap_order, gold_node_order, gold_values = reverse_insert10_heap_structure

    gold_node_order2 = numpy.full((20,), -1, dtype=numpy.int32)
    gold_heap_order2 = numpy.full((20,), -1, dtype=numpy.int32)
    gold_values2 = numpy.full((20,), -1, dtype=numpy.int32)
    gold_node_order2[node_ind2] = gold_node_order
    gold_heap_order2[:10] = 2 * gold_heap_order + 1
    gold_values2[:10] = gold_values

    numpy.testing.assert_equal(gold_node_order2, node_order.numpy())
    numpy.testing.assert_equal(gold_heap_order2, heap_order.numpy())
    numpy.testing.assert_equal(gold_values2, values.numpy())
