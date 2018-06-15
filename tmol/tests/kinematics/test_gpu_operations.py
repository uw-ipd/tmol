import numpy
import torch
from numba import cuda
import time

from tmol.kinematics import (
    backwardKin,
    forwardKin,
)

import tmol.kinematics.gpu_operations
from tmol.kinematics.datatypes import RefoldData
from tmol.kinematics.builder import KinematicBuilder
from tmol.tests.kinematics.test_torch_op import gradcheck_test_system

from tmol.kinematics.datatypes import NodeType, KinTree, KinDOF, BondDOF, JumpDOF
from tmol.kinematics.operations import BondTransforms, JumpTransforms, SegScan, Fscollect

from tmol.tests.torch import requires_cuda


def test_gpu_refold_data_construction(ubq_system):
    tsys = ubq_system
    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree
    kincoords = torch.DoubleTensor(tsys.coords[kintree.id])

    natoms, ndepths, ri2ki, ki2ri, parent_ko, non_subpath_parent_ro, \
        branching_factor_ko, subpath_child_ko, \
        subpath_length_ko, is_subpath_root_ko, is_subpath_leaf_ko, \
        refold_atom_depth_ko, refold_atom_range_for_depth, subpath_root_ro, \
        is_leaf_dso, n_nonpath_children_ko, \
        derivsum_path_depth_ko, derivsum_atom_range_for_depth, ki2dsi, dsi2ki, \
        non_path_children_ko, non_path_children_dso = \
        tmol.kinematics.gpu_operations.construct_refold_and_derivsum_orderings(kintree)

    for ii_ki in range(natoms):
        parent_ki = parent_ko[ii_ki]
        ii_ri = ki2ri[ii_ki]
        parent_ri = ki2ri[parent_ki]
        assert parent_ki == ii_ki or \
            non_subpath_parent_ro[ii_ri] == -1 or \
            non_subpath_parent_ro[ii_ri] == parent_ri

        child_ki = subpath_child_ko[ii_ki]
        assert child_ki == -1 or \
            non_subpath_parent_ro[ki2ri[child_ki]] == -1

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
            #print(ii,ii_ki,jj,"child",child,"child dsi",refold_data.dsi2ki[child],refold_data.parent_ko[refold_data.dsi2ki[child]])
            assert child == -1 or ii_ki == parent_ko[dsi2ki[child]]


@requires_cuda
def test_gpu_refold_ordering(ubq_system):

    #numpy.set_printoptions(threshold=numpy.nan, precision=3)

    #kintree, dof_metadata, kincoords = gradcheck_test_system

    tsys = ubq_system
    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree
    kincoords = torch.DoubleTensor(tsys.coords[kintree.id])

    refold_data = tmol.kinematics.gpu_operations.refold_data_from_kintree(
        kintree, torch.device("cuda")
    )

    dofs = backwardKin(kintree, kincoords).dofs

    # 1) local HTs
    HTs = torch.empty([refold_data.natoms, 4, 4], dtype=torch.double)

    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0
    HTs[0] = torch.eye(4)

    bondSelector = kintree.doftype == NodeType.bond
    HTs[bondSelector] = BondTransforms(dofs.bond[bondSelector])

    jumpSelector = kintree.doftype == NodeType.jump
    HTs[jumpSelector] = JumpTransforms(dofs.jump[jumpSelector])

    HTs_d = tmol.kinematics.gpu_operations.get_devicendarray(HTs)

    tmol.kinematics.gpu_operations.segscan_hts_gpu(HTs_d, refold_data)

    #HTs = HTs_d.copy_to_host()
    refold_kincoords = HTs.numpy()[:, :3, 3].copy()

    # needed for ubq_system, but not gradcheck_test_system:
    refold_kincoords[0, :] = numpy.nan

    numpy.testing.assert_allclose(kincoords, refold_kincoords, 1e-4)

    # Timing
    #start_time = time.time()
    #for i in range(10000):
    #    tmol.kinematics.datatypes.segscan_hts_gpu(HTs_d, refold_data)
    #
    #print(
    #    "--- refold %f seconds ---" % ((time.time() - start_time) / 10000)
    #)

    # ok, now, let's see that f1f2 summation is functioning properly
    f1s = torch.arange(
        refold_data.natoms * 3, dtype=torch.float64
    ).reshape((refold_data.natoms, 3)) / 512.
    f2s = torch.arange(
        refold_data.natoms * 3, dtype=torch.float64
    ).reshape((refold_data.natoms, 3)) / 512.

    f1f2s = numpy.zeros((refold_data.natoms, 6))
    f1f2s[:, 0:3] = f1s
    f1f2s[:, 3:6] = f2s

    SegScan(f1s, kintree.parent, Fscollect, True)
    SegScan(f2s, kintree.parent, Fscollect, True)

    f1f2s_d = cuda.to_device(f1f2s)
    tmol.kinematics.gpu_operations.segscan_f1f2s_gpu(f1f2s_d, refold_data)

    f1f2s = f1f2s_d.copy_to_host()

    f1f2s_gold = numpy.concatenate((f1s, f2s), axis=1)

    # clear the 0th entry; its contents are garbage
    f1f2s_gold[0, :] = 0
    f1f2s[0, :] = 0

    numpy.testing.assert_allclose(f1f2s_gold, f1f2s, 1e-4)


@requires_cuda
def test_gpu_segscan2(ubq_system):

    numpy.set_printoptions(threshold=numpy.nan, precision=3)

    #kintree, dof_metadata, kincoords = gradcheck_test_system

    tsys = ubq_system
    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree
    kincoords = torch.DoubleTensor(tsys.coords[kintree.id])

    refold_data = tmol.kinematics.gpu_operations.refold_data_from_kintree(
        kintree, torch.device("cuda")
    )

    dofs = backwardKin(kintree, kincoords).dofs

    # 1) local HTs
    HTs = torch.empty([refold_data.natoms, 4, 4], dtype=torch.double)

    assert kintree.doftype[0] == NodeType.root
    assert kintree.parent[0] == 0
    HTs[0] = torch.eye(4)

    bondSelector = kintree.doftype == NodeType.bond
    HTs[bondSelector] = BondTransforms(dofs.bond[bondSelector])

    jumpSelector = kintree.doftype == NodeType.jump
    HTs[jumpSelector] = JumpTransforms(dofs.jump[jumpSelector])

    HTs_d = tmol.kinematics.gpu_operations.get_devicendarray(HTs)

    tmol.kinematics.gpu_operations.segscan_hts_gpu2(HTs_d, refold_data)

    #HTs = HTs_d.copy_to_host()
    refold_kincoords = HTs.numpy()[:, :3, 3].copy()

    # needed for ubq_system, but not gradcheck_test_system:
    refold_kincoords[0, :] = numpy.nan

    ki2ri = refold_data.ki2ri_d.copy_to_host()
    ri2ki = ki2ri.copy()
    for i in range(ki2ri.shape[0]):
        ri2ki[ki2ri[i]] = i

    #for i in range(kincoords.shape[0]):
    #    print(i, ri2ki[i], kincoords[ri2ki[i],:].numpy() - refold_kincoords[ri2ki[i],:])
    #print("max diff:", numpy.max(numpy.abs(kincoords[1:,:].numpy() - refold_kincoords[1:,:])))

    numpy.testing.assert_allclose(kincoords, refold_kincoords, 1e-4)

    # # Timing
    # start_time = time.time()
    # for i in range(1000):
    #     tmol.kinematics.gpu_operations.segscan_hts_gpu(HTs_d, refold_data)
    #
    # print("--- refold %f seconds ---" % ((time.time() - start_time) / 1000))
