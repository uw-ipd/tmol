import pytest

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

from tmol.kinematics.builder import KinematicBuilder

import tmol.kinematics.cpu_operations as cpu_operations
import tmol.kinematics.gpu_operations as gpu_operations


@pytest.fixture
def target_device(numba_cuda_or_cudasim):
    # Reload jit modules to ensure that current cuda execution environment, set
    # by fixture, is active for jit functions.
    importlib.reload(cpu_operations.jit)
    importlib.reload(gpu_operations.derivsum_jit)
    importlib.reload(gpu_operations.forward_jit)
    importlib.reload(gpu_operations.scan_paths_jit)

    if numba_cuda_or_cudasim.cudadrv == numba.cuda.simulator.cudadrv:
        return "cpu"
    else:
        return "cuda"


@pytest.fixture
def target_system(target_device, min_system, ubq_system):
    """min system in cudasim tests to reduce test runtime"""
    return {
        "cpu": min_system,
        "cuda": ubq_system,
    }[target_device]


@pytest.fixture
def target_kintree(target_system):
    return KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, target_system.bonds)
    ).kintree


@pytest.mark.benchmark(group="score_setup")
def test_gpu_refold_data_construction(benchmark, ubq_system):
    kintree = target_kintree(ubq_system)

    @benchmark
    def tree_reordering() -> GPUKinTreeReordering:
        return GPUKinTreeReordering.calculate_from_kintree(kintree)

    o = tree_reordering

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


@pytest.mark.benchmark(group="kinematic_op_micro")
def test_parallel_and_iterative_refold(
        benchmark, target_system, target_kintree, target_device
):
    coords = torch.tensor(target_system.coords[target_kintree.id]
                          ).to(device=target_device)
    kintree = target_kintree.to(device=target_device)

    bkin = backwardKin(target_kintree, coords)

    local_hts = DOFTransforms(kintree.doftype, bkin.dofs)

    ### refold from local hts should equal global hts from backward kinematics
    ### inplace operations should produce same result as non-inplace
    # iterative case

    iterative_refold_hts = iterative_refold(
        local_hts.cpu(), kintree.parent.cpu(), inplace=False
    )
    numpy.testing.assert_array_almost_equal(bkin.hts, iterative_refold_hts)

    iterative_refold_hts_inplace = local_hts.clone().cpu()
    iterative_refold(
        iterative_refold_hts_inplace, kintree.parent.cpu(), inplace=True
    )
    numpy.testing.assert_array_almost_equal(
        iterative_refold_hts, iterative_refold_hts_inplace
    )

    # parallel case
    refold_ordering = GPUKinTreeReordering.for_kintree(kintree).refold_ordering

    @benchmark
    def parallel_refold_hts():
        result = refold_ordering.segscan_hts(local_hts, inplace=False)
        numba.cuda.synchronize()
        return result

    numpy.testing.assert_array_almost_equal(bkin.hts, parallel_refold_hts)

    parallel_refold_hts_inplace = local_hts.clone()
    (
        GPUKinTreeReordering.for_kintree(kintree).refold_ordering.segscan_hts(
            parallel_refold_hts_inplace, inplace=True
        )
    )
    numpy.testing.assert_array_almost_equal(
        parallel_refold_hts, parallel_refold_hts_inplace
    )


@pytest.mark.benchmark(group="kinematic_op_micro")
def test_parallel_and_iterative_derivsum(
        benchmark, target_system, target_kintree, target_device
):
    coords = torch.tensor(target_system.coords[target_kintree.id]
                          ).to(device=target_device)
    kintree = target_kintree.to(device=target_device)

    torch.manual_seed(1663)
    dsc_dx = (torch.rand_like(coords) * 2) - 1

    f1s = torch.cross(coords, coords - dsc_dx)
    f2s = dsc_dx.clone()  # clone input buffer before aggregation

    f1f2s = torch.cat((f1s, f2s), 1)

    ### deriv summation sould be equivalent in both interative and parallel mode

    iterative_f1f2_sums = cpu_operations.iterative_f1f2_summation(
        f1f2s.cpu(), kintree.parent, inplace=False
    )

    # Load and cache ordering for benchmark
    derivsum_ordering = GPUKinTreeReordering.for_kintree(
        kintree
    ).derivsum_ordering

    @benchmark
    def parallel_f1f2_sums():
        result = derivsum_ordering.segscan_f1f2s(f1f2s, inplace=False)
        numba.cuda.synchronize()
        return result

    numpy.testing.assert_array_almost_equal(
        iterative_f1f2_sums, parallel_f1f2_sums
    )

    ### inplace operations should produce same result as non-inplace
    iterative_f1f2_sums_inplace = f1f2s.clone().cpu()
    cpu_operations.iterative_f1f2_summation(
        iterative_f1f2_sums_inplace, kintree.parent, inplace=True
    )
    numpy.testing.assert_array_almost_equal(
        iterative_f1f2_sums, iterative_f1f2_sums_inplace
    )

    parallel_f1f2_sums_inplace = f1f2s.clone()
    (
        GPUKinTreeReordering.for_kintree(kintree)
        .derivsum_ordering.segscan_f1f2s(
            parallel_f1f2_sums_inplace, inplace=True
        )
    )
    numpy.testing.assert_array_almost_equal(
        parallel_f1f2_sums,
        parallel_f1f2_sums_inplace,
    )
