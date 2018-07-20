import pytest

import numpy
import torch
import numba

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

from tmol.tests.torch import requires_cuda

CONSISTENCY_CHECK_NITER = 10


def system_kintree(target_system):
    return KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, target_system.bonds)
    ).kintree


@pytest.mark.benchmark(group="score_setup")
def test_gpu_refold_data_construction(benchmark, ubq_system):
    kintree = system_kintree(ubq_system)

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


@requires_cuda
@pytest.mark.benchmark(group="kinematic_op_micro_refold")
@pytest.mark.parametrize(
    "segscan_num_threads",
    [
        128,
        256,
        # block size of 512  exceeds shared memory limit
        pytest.param(512, marks=pytest.mark.xfail),
    ]
)
def test_refold_values(benchmark, big_system, segscan_num_threads):
    target_device = torch.device("cuda")
    target_kintree = system_kintree(big_system)

    coords = torch.tensor(big_system.coords[target_kintree.id]
                          ).to(device=target_device)
    kintree = target_kintree.to(device=target_device)

    bkin = backwardKin(target_kintree, coords)

    local_hts = DOFTransforms(kintree.doftype, bkin.dofs)

    ### Iterative Case
    # values should match generated hts
    # inplace operations should produce same result as non-inplace
    iterative_refold_hts = iterative_refold(
        local_hts.cpu(), kintree.parent.cpu(), inplace=False
    )

    iterative_refold_hts_inplace = local_hts.clone().cpu()
    iterative_refold(
        iterative_refold_hts_inplace, kintree.parent.cpu(), inplace=True
    )

    numpy.testing.assert_allclose(bkin.hts, iterative_refold_hts, atol=1e-6)
    numpy.testing.assert_allclose(
        iterative_refold_hts, iterative_refold_hts_inplace, atol=1e-6
    )

    ### Parallel Case
    # values should match generated hts
    # inplace operations should produce same result as non-inplace
    refold_ordering = GPUKinTreeReordering.for_kintree(kintree).refold_ordering

    # perform single run here *before* adjusting the thread block size
    parallel_refold_hts_inplace = local_hts.clone()
    (
        GPUKinTreeReordering.for_kintree(kintree).refold_ordering.segscan_hts(
            parallel_refold_hts_inplace, inplace=True
        )
    )

    # override the segscan_num_threads on otherwise frozen object
    object.__setattr__(
        refold_ordering, "segscan_num_threads", segscan_num_threads
    )

    @benchmark
    def parallel_refold_hts():
        result = refold_ordering.segscan_hts(local_hts, inplace=False)
        numba.cuda.synchronize()
        return result

    numpy.testing.assert_allclose(bkin.hts, parallel_refold_hts, atol=1e-6)
    numpy.testing.assert_allclose(
        parallel_refold_hts, parallel_refold_hts_inplace, atol=1e-6
    )

    # results must be consistent over repeated invocations, tests issue 90
    niter = CONSISTENCY_CHECK_NITER
    repeated_parallel_results = torch.cat([
        refold_ordering.segscan_hts(local_hts, inplace=False)[None, ...]
        for _ in range(niter)
    ])

    numpy.testing.assert_allclose(
        repeated_parallel_results,
        torch.cat([bkin.hts[None, ...] for _ in range(niter)]),
        atol=1e-6
    )


@requires_cuda
@pytest.mark.benchmark(group="kinematic_op_micro_derivsum")
@pytest.mark.parametrize("segscan_num_threads", [128, 256, 512])
def test_derivsum_values(benchmark, big_system, segscan_num_threads):
    target_device = torch.device("cuda")
    target_kintree = system_kintree(big_system)

    coords = torch.tensor(big_system.coords[target_kintree.id]
                          ).to(device=target_device)
    kintree = target_kintree.to(device=target_device)

    torch.manual_seed(1663)
    dsc_dx = (torch.rand_like(coords) * 2) - 1

    f1s = torch.cross(coords, coords - dsc_dx)
    f2s = dsc_dx.clone()  # clone input buffer before aggregation

    f1f2s = torch.cat((f1s, f2s), 1)

    ### iterative case
    # inplace should match non-inplace operations

    iterative_f1f2_sums = cpu_operations.iterative_f1f2_summation(
        f1f2s.cpu(), kintree.parent, inplace=False
    )

    iterative_f1f2_sums_inplace = f1f2s.clone().cpu()
    cpu_operations.iterative_f1f2_summation(
        iterative_f1f2_sums_inplace, kintree.parent, inplace=True
    )

    numpy.testing.assert_array_almost_equal(
        iterative_f1f2_sums, iterative_f1f2_sums_inplace
    )

    ### Parallel case

    # Load and cache ordering for benchmark
    derivsum_ordering = GPUKinTreeReordering.for_kintree(
        kintree
    ).derivsum_ordering

    # perform single run here *before* adjusting the thread block size
    parallel_f1f2_sums_inplace = f1f2s.clone()
    (
        GPUKinTreeReordering.for_kintree(kintree)
        .derivsum_ordering.segscan_f1f2s(
            parallel_f1f2_sums_inplace, inplace=True
        )
    )

    # override the segscan_num_threads on otherwise frozen object
    object.__setattr__(
        derivsum_ordering, "segscan_num_threads", segscan_num_threads
    )

    @benchmark
    def parallel_f1f2_sums():
        result = derivsum_ordering.segscan_f1f2s(f1f2s, inplace=False)
        numba.cuda.synchronize()
        return result

    # deriv summation sould be equivalent in both interative and parallel mode
    # (can't compare to "standard" value as in refold case

    numpy.testing.assert_allclose(iterative_f1f2_sums, parallel_f1f2_sums)

    # inplace should match non-inplace operations
    numpy.testing.assert_allclose(
        parallel_f1f2_sums,
        parallel_f1f2_sums_inplace,
    )

    # results must be consistent over repeated invocations, tests issue 90
    niter = CONSISTENCY_CHECK_NITER
    repeated_parallel_results = torch.cat([
        derivsum_ordering.segscan_f1f2s(f1f2s, inplace=False)[None, ...]
        for _ in range(niter)
    ])
    repeated_iterative_results = torch.cat([
        iterative_f1f2_sums[None, ...] for _ in range(niter)
    ])

    numpy.testing.assert_allclose(
        repeated_parallel_results,
        repeated_iterative_results,
    )
