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


def test_builder_refold():
    from tmol.tests.data.pdb import data
    from tmol.system.residue.io import read_pdb

    tsys = read_pdb(data["1ubq"])

    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree

    #fd array was 1229 x 1 x 3 but kin code expects N x 3
    kincoords = torch.tensor(tsys.coords[kintree.id]).squeeze()
    refold_kincoords = forwardKin(
        kintree,
        backwardKin(kintree, kincoords).dofs
    ).coords

    assert numpy.all(refold_kincoords[0] == 0)

    refold_coords = numpy.full_like(tsys.coords, numpy.nan)
    refold_coords[kintree.id[1:].squeeze()] = refold_kincoords[1:]

    numpy.savetxt("tsys.csv", tsys.coords, delimiter=",")
    numpy.savetxt("refold.csv", refold_coords, delimiter=",")

    numpy.testing.assert_allclose(tsys.coords, refold_coords)


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


def test_build():
    from tmol.tests.data.pdb import data
    from tmol.system.residue.io import read_pdb

    tsys = read_pdb(data["1ubq"])

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
