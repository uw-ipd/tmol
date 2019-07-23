import pytest
import toolz

import scipy.sparse.csgraph as csgraph

import numpy
import torch

from tmol.score.bonded_atom import BondedAtomScoreGraph
from tmol.system.packed import PackedResidueSystem, PackedResidueSystemStack


def test_bonded_atom_clone_factory(ubq_system: PackedResidueSystem):
    src: BondedAtomScoreGraph = BondedAtomScoreGraph.build_for(ubq_system)

    # Bond graph is referenced
    clone = BondedAtomScoreGraph.build_for(src)
    assert clone.bonds is src.bonds
    numpy.testing.assert_allclose(src.bonds, clone.bonds)
    numpy.testing.assert_allclose(src.bonded_path_length, clone.bonded_path_length)

    clone.bonds = clone.bonds[: len(clone.bonds) // 2]
    assert clone.bonds is not src.bonds
    with pytest.raises(AssertionError):
        numpy.testing.assert_allclose(src.bonds, clone.bonds)
    with pytest.raises(AssertionError):
        numpy.testing.assert_allclose(src.bonded_path_length, clone.bonded_path_length)

    # Atom types are referenced
    assert clone.atom_types is src.atom_types


def test_bonded_atom_clone_factory_from_stacked_systems(
    ubq_system: PackedResidueSystem
):
    twoubq = PackedResidueSystemStack((ubq_system, ubq_system))
    basg = BondedAtomScoreGraph.build_for(twoubq)

    assert basg.atom_types.shape == (2, basg.system_size)
    assert basg.atom_names.shape == (2, basg.system_size)
    assert basg.res_names.shape == (2, basg.system_size)
    assert basg.res_indices.shape == (2, basg.system_size)


def test_real_atoms(ubq_system: PackedResidueSystem):
    """``real_atoms`` is set for every residue's atom in an packed residue system."""
    expected_real_indices = list(
        toolz.concat(
            range(i, i + len(r.coords))
            for i, r in zip(ubq_system.res_start_ind, ubq_system.residues)
        )
    )

    src: BondedAtomScoreGraph = BondedAtomScoreGraph.build_for(ubq_system)

    assert src.real_atoms.shape == (1, src.system_size)
    assert list(numpy.flatnonzero(numpy.array(src.real_atoms))) == expected_real_indices


def test_bonded_path_length(ubq_system: PackedResidueSystem):
    """Bonded path length is evaluated up to MAX_BONDED_PATH_LENGTH."""

    src: BondedAtomScoreGraph = BondedAtomScoreGraph.build_for(ubq_system)
    src_bond_table = numpy.zeros((src.system_size, src.system_size))
    src_bond_table[src.bonds[:, 1], src.bonds[:, 2]] = 1
    bond_graph = csgraph.csgraph_from_dense(src_bond_table)
    distance_table = torch.from_numpy(
        csgraph.shortest_path(bond_graph, directed=False, unweighted=True)
    ).to(torch.float)

    for mlen in (None, 6, 8, 12):
        if mlen is not None:
            src.MAX_BONDED_PATH_LENGTH = mlen

        assert src.bonded_path_length.shape == (1, src.system_size, src.system_size)
        assert numpy.all(
            src.bonded_path_length[0][distance_table > src.MAX_BONDED_PATH_LENGTH]
            == numpy.inf
        )
        assert numpy.all(
            src.bonded_path_length[0][distance_table < src.MAX_BONDED_PATH_LENGTH]
            == distance_table[distance_table < src.MAX_BONDED_PATH_LENGTH]
        )
