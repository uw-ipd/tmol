"""Chemistry already supplied by a reader should not require rereading its CIF."""

import biotite.structure as struc
import numpy as np
import pytest

from tmol.ligand._detect import (
    _component_types_from_annotations,
    polymer_entity_residues,
)


def atoms():
    result = struc.AtomArray(4)
    result.res_name[:] = ["ZZZ", "ZZZ", "CAP", "LIG"]
    result.set_annotation(
        "chem_comp_type",
        np.array(
            ["D-PEPTIDE LINKING", "D-PEPTIDE LINKING", "NON-POLYMER", "NON-POLYMER"]
        ),
    )
    result.set_annotation("is_polymer", np.array([True, True, True, False]))
    return result


def test_annotations_supply_types_and_polymer_membership():
    array = atoms()
    assert _component_types_from_annotations(array) == {
        "ZZZ": "D-PEPTIDE LINKING",
        "CAP": "NON-POLYMER",
        "LIG": "NON-POLYMER",
    }
    assert polymer_entity_residues(array) == frozenset({"ZZZ", "CAP"})
    # A numbering proxy must not reclassify an explicitly non-polymer ligand.
    array.set_annotation("label_seq_id", np.array(["1", "1", "2", "3"]))
    assert polymer_entity_residues(array) == frozenset({"ZZZ", "CAP"})


def test_conflicting_type_annotations_require_explicit_resolution():
    array = atoms()
    array.chem_comp_type[1] = "L-PEPTIDE LINKING"
    with pytest.raises(ValueError, match="Conflicting chem_comp_type.*ZZZ"):
        _component_types_from_annotations(array)
    assert _component_types_from_annotations(array, {"ZZZ": "OTHER"})["ZZZ"] == "OTHER"


def test_unannotated_reader_retains_unknown_membership():
    array = struc.AtomArray(1)
    assert polymer_entity_residues(array) is None
    assert _component_types_from_annotations(array) == {}
