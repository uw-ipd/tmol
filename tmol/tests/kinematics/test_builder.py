import numpy
import torch

from tmol.kinematics import (
    backwardKin,
    forwardKin,
)

import tmol.kinematics.builder
from tmol.kinematics.builder import KinematicBuilder

from tmol.system.residue.packed import PackedResidueSystem
from tmol.types.array import NDArray


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


def report_cut_results(
        sys: PackedResidueSystem,
        connections: NDArray(int)[:, 2],
):
    ctups = set(map(tuple, connections))

    missing_torsions = []

    for ti, t in enumerate(sys.torsion_metadata):
        bp = (t["atom_index_b"], t["atom_index_c"])
        if any(ai == -1 for ai in bp):
            continue

        if not (bp in ctups or tuple(reversed(bp)) in ctups):
            missing_torsions.append(ti)

    missing_bonds = []

    for bi, b in enumerate(sys.bonds):
        if not (tuple(b) in ctups or tuple(reversed(b)) in ctups):
            if b[0] < b[1]:
                missing_bonds.append(bi)

    return {
        "missing_torsions": sys.torsion_metadata[missing_torsions],
        "missing_bonds":
            sys.atom_metadata[sys.bonds[missing_bonds]]
            [["residue_index", "residue_name", "atom_name"]]
    }


def test_build(ubq_system):
    tsys = ubq_system

    torsion_pairs = (
        tsys.torsion_metadata[[
            "atom_index_b",
            "atom_index_c",
        ]].copy().view(int).reshape(-1, 2)
    )

    torsion_bonds = torsion_pairs[numpy.all(torsion_pairs > 0, axis=-1)]

    weighted_bonds = (
        # All entries must be non-zero or sparse graph tools will entries.
        KinematicBuilder.bond_csgraph(tsys.bonds, [-1], tsys.system_size) +
        KinematicBuilder.bond_csgraph(
            torsion_bonds, [-1e-3], tsys.system_size
        )
    )

    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, weighted_bonds)
    ).kintree

    kinematic_connections = tmol.kinematics.builder.kintree_connections(
        kintree
    )
    kinematic_tree_results = report_cut_results(tsys, kinematic_connections)

    assert len(kinematic_tree_results["missing_torsions"]) == 0, (
        f"Generated kinematic tree did not cover all named torsions.\n"
        f"torsions:\n{kinematic_tree_results['missing_torsions']}\n"
        f"bonds:\n{kinematic_tree_results['missing_bonds']}\n"
    )
