import pytest

import numpy

from tmol.score.bonded_atom import BondedAtomScoreGraph


def test_bonded_atom_clone_factory(ubq_system):
    src: BondedAtomScoreGraph = BondedAtomScoreGraph.build_for(ubq_system)

    # Bond graph is referenced
    clone = BondedAtomScoreGraph.build_for(src)
    assert clone.bonds is src.bonds
    numpy.testing.assert_allclose(src.bonds, clone.bonds)
    numpy.testing.assert_allclose(
        src.bonded_path_length, clone.bonded_path_length
    )

    clone.bonds = clone.bonds[:len(clone.bonds) // 2]
    assert clone.bonds is not src.bonds
    with pytest.raises(AssertionError):
        numpy.testing.assert_allclose(src.bonds, clone.bonds)
    with pytest.raises(AssertionError):
        numpy.testing.assert_allclose(
            src.bonded_path_length, clone.bonded_path_length
        )

    # Atom types are referenced
    assert clone.atom_types is src.atom_types
