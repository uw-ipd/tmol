"""Filtering incomplete backbones must not silently remove chemical partners."""

import biotite.structure as struc
import numpy as np
import pytest

from tmol.io._pose_stack_from_biotite import (
    _filter_supported_atoms_and_connectivity,
    canonical_ordering_for_biotite,
)


def cysteine_pair(bond, *, chains=("A", "A"), missing=True):
    array = struc.AtomArray(12)
    array.atom_name = np.tile(["N", "CA", "C", "O", "CB", "SG"], 2)
    array.element = np.tile(["N", "C", "C", "O", "C", "S"], 2)
    array.res_name[:] = "CYS"
    array.res_id = np.repeat([1, 2], 6)
    array.chain_id = np.repeat(chains, 6)
    array.coord = np.arange(36, dtype=np.float32).reshape(12, 3)
    if missing:
        array.coord[0] = np.nan
    array.bonds = struc.BondList(12, np.array([[*bond, struc.BondType.SINGLE]]))
    return array


@pytest.mark.parametrize(
    "bond,chains",
    [
        ((5, 11), ("A", "A")),  # Declared disulfide.
        ((4, 11), ("A", "A")),  # Other sidechain crosslink.
        ((0, 8), ("A", "A")),  # Reverse backbone link closes a cycle.
        ((2, 6), ("A", "B")),  # Explicit polymer link across chains.
    ],
)
def test_reject_filtering_one_end_of_declared_chemical_link(bond, chains):
    array = cysteine_pair(bond, chains=chains)
    before = array.copy()
    with pytest.raises(ValueError, match="discard.*covalent"):
        _filter_supported_atoms_and_connectivity(
            array, canonical_ordering_for_biotite()
        )
    np.testing.assert_equal(array.coord, before.coord)
    np.testing.assert_array_equal(array.bonds.as_array(), before.bonds.as_array())


def test_reject_when_one_model_lacks_the_covalent_partner_backbone():
    complete = cysteine_pair((5, 11), missing=False)
    incomplete = cysteine_pair((5, 11))
    stack = struc.stack([complete, incomplete])
    with pytest.raises(ValueError, match="discard.*covalent"):
        _filter_supported_atoms_and_connectivity(
            stack, canonical_ordering_for_biotite()
        )


def test_ordinary_sequential_backbone_gap_remains_supported():
    array = cysteine_pair((2, 6))
    kept, breaks = _filter_supported_atoms_and_connectivity(
        array, canonical_ordering_for_biotite()
    )
    np.testing.assert_array_equal(kept.res_id, np.full(6, 2))
    assert breaks.shape == (1, 2)


def test_complete_crosslink_keeps_both_residues_and_its_bond():
    array = cysteine_pair((5, 11), missing=False)
    kept, _ = _filter_supported_atoms_and_connectivity(
        array, canonical_ordering_for_biotite()
    )
    np.testing.assert_array_equal(kept.coord, array.coord)
    np.testing.assert_array_equal(kept.bonds.as_array(), array.bonds.as_array())
