import numpy
import torch
from numba import cuda

import tmol.kinematics.gpu_operations
from tmol.kinematics.builder import KinematicBuilder

from tmol.kinematics.datatypes import NodeType
from tmol.kinematics.operations import (
    backwardKin, BondTransforms, JumpTransforms, ExecutionStrategy,
    GPUKinTreeReordering, SegScanDerivs
)

from tmol.tests.torch import requires_cuda


@requires_cuda
def test_gpu_refold_data_construction(ubq_system):
    tsys = ubq_system
    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree
    torch.DoubleTensor(tsys.coords[kintree.id])

    ordering = GPUKinTreeReordering.from_kintree(kintree, torch.device("cuda"))

    # Extract path data from tree reordering.
    natoms = ordering.natoms
    subpath_child_ko = ordering.subpath_child_ko
    ki2ri = ordering.ki2ri_d.copy_to_host()
    dsi2ki = ordering.dsi2ki
    parent_ko = kintree.parent
    non_subpath_parent_ro = ordering.non_subpath_parent_ro_d.copy_to_host()
    subpath_child_ko = ordering.subpath_child_ko
    non_path_children_ko = ordering.non_path_children_ko
    non_path_children_dso = ordering.non_path_children_dso_d.copy_to_host()

    for ii_ki in range(natoms):
        parent_ki = kintree.parent[ii_ki]

        ii_ri = ki2ri[ii_ki]
        parent_ri = ki2ri[parent_ki]
        assert parent_ki == ii_ki or \
            non_subpath_parent_ro[ii_ri] == -1 or \
            non_subpath_parent_ro[ii_ri] == parent_ri

        child_ki = subpath_child_ko[ii_ki]
        assert child_ki == -1 or non_subpath_parent_ro[ki2ri[child_ki]] == -1

    for ii in range(natoms):
        for jj in range(non_path_children_ko.shape[1]):
            child = non_path_children_ko[ii, jj]
            assert child == -1 or parent_ko[child] == ii
        first_child = subpath_child_ko[ii]
        assert first_child == -1 or parent_ko[first_child] == ii

    for ii in range(natoms):
        for jj in range(non_path_children_ko.shape[1]):
            child = non_path_children_dso[ii, jj]
            ii_ki = dsi2ki[ii]
            assert child == -1 or ii_ki == parent_ko[dsi2ki[child]]


@requires_cuda
def test_gpu_refold_ordering(ubq_system):

    tsys = ubq_system
    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree
    kincoords = torch.DoubleTensor(tsys.coords[kintree.id])

    reordering = GPUKinTreeReordering.from_kintree(
        kintree, torch.device("cuda")
    )

    dofs = backwardKin(kintree, kincoords).dofs

    # 1) local HTs
    HTs = torch.empty([reordering.natoms, 4, 4], dtype=torch.double)

    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0
    HTs[0] = torch.eye(4)

    bondSelector = kintree.doftype == NodeType.bond
    HTs[bondSelector] = BondTransforms(dofs.bond[bondSelector])

    jumpSelector = kintree.doftype == NodeType.jump
    HTs[jumpSelector] = JumpTransforms(dofs.jump[jumpSelector])

    # temp
    HTs_d = cuda.to_device(HTs.numpy())
    #HTs_d = tmol.kinematics.gpu_operations.get_devicendarray(HTs)

    tmol.kinematics.gpu_operations.segscan_hts_gpu(HTs_d, reordering)

    HTs = HTs_d.copy_to_host()
    refold_kincoords = HTs[:, :3, 3].copy()
    # temp refold_kincoords = HTs.numpy()[:, :3, 3].copy()

    # needed for ubq_system, but not gradcheck_test_system:
    refold_kincoords[0, :] = numpy.nan

    numpy.testing.assert_allclose(kincoords, refold_kincoords, 1e-4)

    # Timing
    #import time
    #start_time = time.time()
    #for i in range(10000):
    #    tmol.kinematics.gpu_operations.segscan_hts_gpu(HTs_d, reordering)
    #
    #print("--- refold %f seconds ---" % ((time.time() - start_time) / 10000))

    # ok, now, let's see that f1f2 summation is functioning properly
    f1s = torch.arange(
        reordering.natoms * 3, dtype=torch.float64
    ).reshape((reordering.natoms, 3)) / 512.
    f2s = torch.arange(
        reordering.natoms * 3, dtype=torch.float64
    ).reshape((reordering.natoms, 3)) / 512.

    f1f2s = numpy.zeros((reordering.natoms, 6))
    f1f2s[:, 0:3] = f1s
    f1f2s[:, 3:6] = f2s

    SegScanDerivs(
        reordering, f1s, f2s, kintree.parent, True,
        ExecutionStrategy.torch_efficient
    )

    f1f2s_d = cuda.to_device(f1f2s)
    tmol.kinematics.gpu_operations.segscan_f1f2s_gpu(f1f2s_d, reordering)

    f1f2s = f1f2s_d.copy_to_host()

    f1f2s_gold = numpy.concatenate((f1s, f2s), axis=1)

    # clear the 0th entry; its contents are garbage
    f1f2s_gold[0, :] = 0
    f1f2s[0, :] = 0

    numpy.testing.assert_allclose(f1f2s_gold, f1f2s, 1e-4)


@requires_cuda
def test_warp_synchronous_gpu_segscan(ubq_system):

    numpy.set_printoptions(threshold=numpy.nan, precision=3)

    #kintree, dof_metadata, kincoords = gradcheck_test_system

    tsys = ubq_system
    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree
    kincoords = torch.DoubleTensor(tsys.coords[kintree.id])

    reordering = GPUKinTreeReordering.from_kintree(
        kintree, torch.device("cuda")
    )

    dofs = backwardKin(kintree, kincoords).dofs

    # 1) local HTs
    HTs = torch.empty([reordering.natoms, 4, 4], dtype=torch.double)

    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0
    HTs[0] = torch.eye(4)

    bondSelector = kintree.doftype == NodeType.bond
    HTs[bondSelector] = BondTransforms(dofs.bond[bondSelector])

    jumpSelector = kintree.doftype == NodeType.jump
    HTs[jumpSelector] = JumpTransforms(dofs.jump[jumpSelector])

    HTs_d = tmol.kinematics.gpu_operations.get_devicendarray(HTs)

    tmol.kinematics.gpu_operations.warp_synchronous_segscan_hts_gpu(
        HTs_d, reordering
    )

    refold_kincoords = HTs.numpy()[:, :3, 3].copy()

    # needed for ubq_system, but not gradcheck_test_system:
    refold_kincoords[0, :] = numpy.nan

    ki2ri = reordering.ki2ri_d.copy_to_host()
    ri2ki = ki2ri.copy()
    for i in range(ki2ri.shape[0]):
        ri2ki[ki2ri[i]] = i

    numpy.testing.assert_allclose(kincoords, refold_kincoords, 1e-4)

    # # Timing
    # import time
    # start_time = time.time()
    # for i in range(1000):
    #     tmol.kinematics.gpu_operations.segscan_hts_gpu(HTs_d, reordering)
    #
    # print("--- refold %f seconds ---" % ((time.time() - start_time) / 1000))
