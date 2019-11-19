import numpy
import torch
import pandas

from tmol.kinematics.operations import inverseKin, forwardKin
from tmol.kinematics.builder import KinematicBuilder
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack
from tmol.score.bonded_atom import BondedAtomScoreGraph


def test_builder_refold(ubq_system):
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

def test_build_two_system_kinematics(ubq_system):
    natoms = numpy.sum(numpy.logical_not(numpy.isnan(ubq_system.coords[:,0])))
    
    dev_cpu = torch.device("cpu") # temp
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    bonds = BondedAtomScoreGraph.build_for(twoubq, device=dev_cpu)
    tworoots = numpy.array((0, twoubq.systems[0].system_size), dtype=int)
    
    ids, parents = KinematicBuilder.bonds_to_connected_component(
        roots=tworoots,
        bonds=bonds.bonds,
        system_size=int(twoubq.systems[0].system_size),
    )

    id_index = pandas.Index(ids)
    root_index = id_index.get_indexer(tworoots)

    builder = KinematicBuilder()
    builder2 = builder.append_connected_components(
        roots=tworoots,
        ids=ids,
        parent_ids=parents,
    )

    tree = builder2.kintree
    assert tree.id.shape[0] == 2 * natoms + 1
    assert tree.id[1] == 0
    assert tree.parent[1 + root_index[0]] == 0
    assert tree.parent[1 + root_index[1]] == 0

