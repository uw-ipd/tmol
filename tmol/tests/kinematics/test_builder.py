import numpy
import torch
from numba import cuda
import time

from tmol.kinematics import (
    backwardKin,
    forwardKin,
)

import tmol.kinematics.datatypes
from tmol.kinematics.datatypes import RefoldData
from tmol.kinematics.builder import KinematicBuilder
from tmol.tests.kinematics.test_torch_op import gradcheck_test_system


def test_builder_refold(ubq_system):
    tsys = ubq_system

    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree

    kincoords = torch.DoubleTensor(tsys.coords[kintree.id])
    refold_kincoords = forwardKin(
        kintree,
        backwardKin(kintree, kincoords).dofs
    ).coords

    assert numpy.all(refold_kincoords[0] == 0)

    refold_coords = numpy.full_like(tsys.coords, numpy.nan)
    refold_coords[kintree.id[1:].squeeze()] = refold_kincoords[1:]

    numpy.testing.assert_allclose(tsys.coords, refold_coords)


def test_builder_framing(ubq_system):
    """Test first-three-atom framing logic in kinematic builder."""
    tsys = ubq_system
    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree

    # The first entries in the tree should be the global DOF root, self-parented,
    # followed by the first atom.
    root_children = kintree[kintree.parent == 0]
    assert len(root_children) == 2
    numpy.testing.assert_array_equal(kintree.parent[:2], [0, 0])
    numpy.testing.assert_array_equal(kintree.id[:2], [-1, 0])

    # The first atom has two children. The first atom and its first child are framed by
    # [first_child, root, second_child]
    atom_root_children = numpy.flatnonzero(numpy.array(kintree.parent) == 1)
    assert len(atom_root_children) == 2

    first_atom = kintree[1]
    assert int(first_atom.frame_x) == atom_root_children[0]
    assert int(first_atom.frame_y) == 1
    assert int(first_atom.frame_z) == atom_root_children[1]

    first_atom_first_child = kintree[atom_root_children[0]]
    assert int(first_atom_first_child.frame_x) == atom_root_children[0]
    assert int(first_atom_first_child.frame_y) == 1
    assert int(first_atom_first_child.frame_z) == atom_root_children[1]

    # The rest of the children are framed by:
    # [self, root, first_child]
    for c in atom_root_children[1:]:
        first_atom_other_child = kintree[c]
        assert int(first_atom_other_child.frame_x) == c
        assert int(first_atom_other_child.frame_y) == 1
        assert int(first_atom_other_child.frame_z) == atom_root_children[0]

    # Other atoms are framed normally, [self, parent, grandparent]
    normal_atoms = numpy.flatnonzero(numpy.array(kintree.parent > 1))
    numpy.testing.assert_array_equal(
        kintree.frame_x[normal_atoms], normal_atoms
    )
    numpy.testing.assert_array_equal(
        kintree.frame_y[normal_atoms], kintree.parent[normal_atoms]
    )
    numpy.testing.assert_array_equal(
        kintree.frame_z[normal_atoms],
        kintree.parent[kintree.parent[normal_atoms]]
    )


def test_gpu_refold_ordering(ubq_system):
    from tmol.kinematics.datatypes import NodeType, KinTree, KinDOF, BondDOF, JumpDOF
    from tmol.kinematics.operations import BondTransforms, JumpTransforms

    numpy.set_printoptions(threshold=numpy.nan, precision=3)

    #kintree, dof_metadata, kincoords = gradcheck_test_system
    tsys = ubq_system
    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree
    kincoords = torch.DoubleTensor(tsys.coords[kintree.id])

    refold_data = RefoldData(kintree.id.shape[0])
    tmol.kinematics.datatypes.determine_refold_indices(kintree, refold_data)
    tmol.kinematics.datatypes.send_refold_data_to_gpu(refold_data)

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

    tmol.kinematics.datatypes.send_refold_data_to_gpu(refold_data)
    if HTs.type() == 'torch.cuda.FloatTensor':
        HTs_d = tmol.kinematics.datatypes.get_devicendarray(HTs)
    else:
        HTs_d = cuda.to_device(HTs.numpy())
        #print("HTs in kintree order");print(HTs.numpy())

    tmol.kinematics.datatypes.segscan_hts_gpu(HTs_d, refold_data)

    if HTs.type() == 'torch.cuda.FloatTensor':
        pass
    else:
        HTs = HTs_d.copy_to_host()
    refold_kincoords = HTs[:, :3, 3].copy()
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
