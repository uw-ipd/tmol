import numpy
import torch

from tmol.kinematics.operations import inverseKin, forwardKin
from tmol.kinematics.builder import KinematicBuilder


def test_builder_refold(ubq_system, torch_device):
    tsys = ubq_system

    kintree = (
        KinematicBuilder()
        .append_connected_component(
            *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
        )
        .kintree
    )

    kincoords = torch.DoubleTensor(tsys.coords[kintree.id])
    dofs = inverseKin(kintree, kincoords)
    refold_kincoords = forwardKin(kintree, dofs)

    assert numpy.all(refold_kincoords[0] == 0)

    refold_coords = numpy.full_like(tsys.coords, numpy.nan)
    refold_coords[kintree.id[1:].squeeze()] = refold_kincoords[1:]

    numpy.testing.assert_allclose(tsys.coords, refold_coords)


def test_builder_framing(ubq_system):
    """Test first-three-atom framing logic in kinematic builder."""
    tsys = ubq_system
    kintree = (
        KinematicBuilder()
        .append_connected_component(
            *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
        )
        .kintree
    )

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
    numpy.testing.assert_array_equal(kintree.frame_x[normal_atoms], normal_atoms)
    numpy.testing.assert_array_equal(
        kintree.frame_y[normal_atoms], kintree.parent[normal_atoms]
    )
    numpy.testing.assert_array_equal(
        kintree.frame_z[normal_atoms],
        kintree.parent[kintree.parent[normal_atoms].to(dtype=torch.long)],
    )
