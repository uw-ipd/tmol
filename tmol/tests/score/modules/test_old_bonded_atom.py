import pytest
import numpy
import torch
import toolz
import scipy.sparse.csgraph as csgraph

from tmol.score.modules.bonded_atom import BondedAtoms
from tmol.score.modules.stacked_system import StackedSystem

from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack

from tmol.score.modules.bases import ScoreSystem


def test_bonded_atom_clone_factory(ubq_system: PackedResidueSystem):
    src: ScoreSystem = ScoreSystem._build_with_modules(ubq_system, {BondedAtoms})

    # Bond graph is referenced
    clone: ScoreSystem = ScoreSystem._build_with_modules(src, {BondedAtoms})

    assert BondedAtoms.get(clone).bonds is BondedAtoms.get(src).bonds
    numpy.testing.assert_allclose(
        BondedAtoms.get(src).bonds, BondedAtoms.get(clone).bonds
    )
    numpy.testing.assert_allclose(
        BondedAtoms.get(src).bonded_path_length,
        BondedAtoms.get(clone).bonded_path_length,
    )

    # Bonded atoms is frozen, can't be modified after init
    with pytest.raises(AttributeError):
        BondedAtoms.get(clone).bonds = BondedAtoms.get(clone).bonds[
            : len(BondedAtoms.get(clone).bonds) // 2
        ]


def test_bonded_atom_clone_factory_from_stacked_systems(
    ubq_system: PackedResidueSystem,
):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    basg: ScoreSystem = ScoreSystem._build_with_modules(twoubq, {BondedAtoms})

    assert BondedAtoms.get(basg).atom_types.shape == (
        2,
        StackedSystem.get(basg).system_size,
    )
    assert BondedAtoms.get(basg).atom_names.shape == (
        2,
        StackedSystem.get(basg).system_size,
    )
    assert BondedAtoms.get(basg).res_names.shape == (
        2,
        StackedSystem.get(basg).system_size,
    )
    assert BondedAtoms.get(basg).res_indices.shape == (
        2,
        StackedSystem.get(basg).system_size,
    )


def test_bonded_path_length(ubq_system: PackedResidueSystem):
    """Bonded path length is evaluated up to MAX_BONDED_PATH_LENGTH."""

    src: ScoreSystem = ScoreSystem._build_with_modules(ubq_system, {BondedAtoms})

    src_bond_table = numpy.zeros(
        (StackedSystem.get(src).system_size, StackedSystem.get(src).system_size)
    )
    src_bond_table[
        BondedAtoms.get(src).bonds[:, 1], BondedAtoms.get(src).bonds[:, 2]
    ] = 1
    bond_graph = csgraph.csgraph_from_dense(src_bond_table)
    distance_table = torch.from_numpy(
        csgraph.shortest_path(bond_graph, directed=False, unweighted=True)
    ).to(torch.float)

    assert BondedAtoms.get(src).bonded_path_length.shape == (
        1,
        StackedSystem.get(src).system_size,
        StackedSystem.get(src).system_size,
    )
    assert (
        BondedAtoms.get(src).bonded_path_length[0][
            distance_table > BondedAtoms.get(src).MAX_BONDED_PATH_LENGTH
        ]
        == numpy.inf
    ).all()
    assert (
        BondedAtoms.get(src).bonded_path_length[0][
            distance_table < BondedAtoms.get(src).MAX_BONDED_PATH_LENGTH
        ]
        == distance_table[distance_table < BondedAtoms.get(src).MAX_BONDED_PATH_LENGTH]
    ).all()

    inds = BondedAtoms.get(src).indexed_bonds
    assert len(inds.bonds.shape) == 3
    assert inds.bonds.shape[2] == 2


def test_real_atoms(ubq_system: PackedResidueSystem):
    """``real_atoms`` is set for every residue's atom in an packed residue system."""
    expected_real_indices = list(
        toolz.concat(
            range(i, i + len(r.coords))
            for i, r in zip(ubq_system.res_start_ind, ubq_system.residues)
        )
    )

    src: ScoreSystem = ScoreSystem._build_with_modules(ubq_system, {BondedAtoms})

    assert BondedAtoms.get(src).real_atoms.shape == (
        1,
        StackedSystem.get(src).system_size,
    )
    assert (
        list(numpy.flatnonzero(numpy.array(BondedAtoms.get(src).real_atoms)))
        == expected_real_indices
    )


def test_variable_bonded_path_length(ubq_res):
    ubq4 = PackedResidueSystem.from_residues(ubq_res[:4])
    ubq6 = PackedResidueSystem.from_residues(ubq_res[:6])
    twoubq = PackedResidueSystemStack((ubq4, ubq6))

    basg_both = ScoreSystem._build_with_modules(twoubq, {BondedAtoms})
    basg4 = ScoreSystem._build_with_modules(ubq4, {BondedAtoms})
    basg6 = ScoreSystem._build_with_modules(ubq6, {BondedAtoms})

    inds_both = BondedAtoms.get(basg_both).indexed_bonds
    inds4 = BondedAtoms.get(basg4).indexed_bonds
    inds6 = BondedAtoms.get(basg6).indexed_bonds

    numpy.testing.assert_allclose(
        inds_both.bonds[0, : inds4.bonds.shape[1]], inds4.bonds[0]
    )
    numpy.testing.assert_allclose(
        inds_both.bonds[1, : inds6.bonds.shape[1]], inds6.bonds[0]
    )

    numpy.testing.assert_allclose(
        inds_both.bond_spans[0, : inds4.bond_spans.shape[1]], inds4.bond_spans[0]
    )
    torch.testing.assert_close(
        inds_both.bond_spans[1, : inds6.bond_spans.shape[1]], inds6.bond_spans[0]
    )
