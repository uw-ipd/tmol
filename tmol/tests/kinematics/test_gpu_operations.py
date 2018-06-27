import numpy
import torch
import numba

import importlib

from tmol.kinematics.operations import (
    DOFTransforms,
    backwardKin,
    iterative_refold,
)
from tmol.kinematics.gpu_operations import (
    GPUKinTreeReordering,
    RefoldOrdering,
    DerivsumOrdering,
    PathPartitioning,
)

import tmol.kinematics.cpu_operations as cpu_operations
import tmol.kinematics.gpu_operations as gpu_operations


def test_gpu_refold_data_construction(ubq_kintree):
    kintree = ubq_kintree

    ### Otherwise test the derived ordering
    o: GPUKinTreeReordering = GPUKinTreeReordering.calculate_from_kintree(
        kintree
    )

    ### Validate path definitions
    sp: PathPartitioning = o.scan_paths
    for ii in range(o.natoms):
        # The subpath child must:
        #   not exist (the node is a leaf)
        #   point back to the node as its parent
        first_child = sp.subpath_child[ii]
        assert first_child == -1 or sp.parent[first_child] == ii

        # Each non-subpath child must:
        #   be non-existant (the node has < max num non-subpath children)
        #   point back to the node as its parent
        for jj in range(sp.nonpath_children.shape[1]):
            child = sp.nonpath_children[ii, jj]
            assert child == -1 or sp.parent[child] == ii

    ### Validate refold ordering
    ro: RefoldOrdering = o.refold_ordering
    for ii_ki in range(o.natoms):
        parent_ki = kintree.parent[ii_ki]

        # The node's parent must:
        #   be "self" (as the node is root?)
        #   be off the subpath (as the node is a subpath root?)
        #   be the kinematic parent (as the node is on a subpath?)
        ii_ri = ro.ki2ri[ii_ki]
        parent_ri = ro.ki2ri[parent_ki]
        assert parent_ki == ii_ki or \
            ro.non_subpath_parent[ii_ri] == -1 or \
            ro.non_subpath_parent[ii_ri] == parent_ri

        # The node's child must:
        #    be non-existant (as the node is a leaf?)
        #    not have a non-subpath parent (as it's parent is this node?)
        child_ki = sp.subpath_child[ii_ki]
        assert (
            child_ki == -1 or ro.non_subpath_parent[ro.ki2ri[child_ki]] == -1
        )

    ### Validate derivsum ordering
    do: DerivsumOrdering = o.derivsum_ordering
    for ii in range(o.natoms):
        # Each non-subpath child must:
        #   be non-existant (the node has < max num non-subpath children)
        #   point back to the node as its parent
        for jj in range(do.nonpath_children.shape[1]):
            child = do.nonpath_children[ii, jj]
            ii_ki = do.dsi2ki[ii]
            assert child == -1 or ii_ki == sp.parent[do.dsi2ki[child]]


def test_parallel_and_iterative_refold(
        ubq_system, ubq_kintree, numba_cuda_or_cudasim
):
    target_device = "cpu" if numba.config.ENABLE_CUDASIM else "cuda"

    # Reload jit modules to ensure that current cuda execution environment, set
    # by fixture, is active for jit functions.
    importlib.reload(cpu_operations.jit)
    importlib.reload(gpu_operations.derivsum_jit)
    importlib.reload(gpu_operations.forward_jit)
    importlib.reload(gpu_operations.scan_paths_jit)

    coords = torch.tensor(ubq_system.coords[ubq_kintree.id]
                          ).to(device=target_device)
    kintree = ubq_kintree.to(device=target_device)

    bkin = backwardKin(ubq_kintree, coords)

    local_hts = DOFTransforms(kintree.doftype, bkin.dofs)

    ### refold from local hts should equal global hts from backward kinematics
    # iterative case

    iterative_refold_hts = iterative_refold(
        local_hts.cpu(), kintree.parent.cpu(), inplace=False
    )
    numpy.testing.assert_array_almost_equal(bkin.hts, iterative_refold_hts)

    # parallel case
    parallel_refold_hts = (
        GPUKinTreeReordering.for_kintree(kintree)
        .refold_ordering.segscan_hts(local_hts, inplace=False)
    )
    numpy.testing.assert_array_almost_equal(bkin.hts, parallel_refold_hts)
