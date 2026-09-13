"""Missing residue charge records must not silently disable electrostatics."""

import attr
import numpy
import pytest

from tmol.database.scoring import PartialCharges
from tmol.pose import PackedBlockTypes
from tmol.score.elec import ElecEnergyTerm, ElecParamResolver


@pytest.fixture
def ala(fresh_default_restype_set):
    return next(
        rt for rt in fresh_default_restype_set.residue_types if rt.name == "ALA"
    )


def change_charges(database, transform):
    return attr.evolve(
        database,
        scoring=attr.evolve(
            database.scoring,
            elec=attr.evolve(
                database.scoring.elec,
                atom_charge_parameters=tuple(
                    transform(database.scoring.elec.atom_charge_parameters)
                ),
            ),
        ),
    )


def without_ala(rows):
    return (r for r in rows if r.res.partition(":")[0] != "ALA")


def test_missing_residue_charges_raise(default_database, ala, torch_device):
    missing = change_charges(default_database, without_ala)
    resolver = ElecParamResolver.from_database(missing.scoring.elec, torch_device)
    with pytest.raises(KeyError, match="Elec charge for atom ALA,N not found"):
        resolver.get_partial_charges_for_block(ala)


@pytest.mark.parametrize("charge", [float("nan"), float("inf"), -float("inf"), 1e100])
def test_used_nonfinite_charge_rejected(default_database, ala, torch_device, charge):
    database = change_charges(
        default_database,
        lambda rows: (
            attr.evolve(r, charge=charge) if r.res == "ALA" and r.atom == "CB" else r
            for r in rows
        ),
    )
    term = ElecEnergyTerm(database, torch_device)
    with pytest.raises(ValueError, match="ALA.*CB.*non-finite"):
        term.setup_block_type(ala)


def test_reused_packed_set_does_not_hide_missing_charges(
    default_database, fresh_default_restype_set, ala, torch_device
):
    packed = PackedBlockTypes.from_restype_list(
        default_database.chemical, fresh_default_restype_set, [ala], torch_device
    )
    valid = ElecEnergyTerm(default_database, torch_device)
    valid.setup_packed_block_types(packed)
    expected = packed.elec_partial_charge.clone()
    missing = change_charges(default_database, without_ala)
    with pytest.raises(KeyError, match="ALA,N not found"):
        ElecEnergyTerm(missing, torch_device).setup_packed_block_types(packed)
    valid.setup_packed_block_types(packed)
    numpy.testing.assert_array_equal(packed.elec_partial_charge.cpu(), expected.cpu())


def test_explicit_zero_charges_and_unused_bad_rows_allowed(
    default_database, ala, torch_device
):
    database = change_charges(
        default_database,
        lambda rows: tuple(without_ala(rows))
        + tuple(PartialCharges("ALA", a.name, 0.0) for a in ala.atoms)
        + (PartialCharges("UNUSED", "X", float("nan")),),
    )
    resolver = ElecParamResolver.from_database(database.scoring.elec, torch_device)
    numpy.testing.assert_array_equal(
        resolver.get_partial_charges_for_block(ala),
        numpy.zeros(len(ala.atoms), dtype=numpy.float32),
    )


def test_default_water_charges_are_explicit(
    default_database, fresh_default_restype_set, torch_device
):
    water = next(
        rt for rt in fresh_default_restype_set.residue_types if rt.name == "HOH"
    )
    rows = {
        r.atom: r.charge
        for r in default_database.scoring.elec.atom_charge_parameters
        if r.res == "HOH"
    }
    assert rows == {"O": 0.0, "H1": 0.0, "H2": 0.0}
    resolver = ElecParamResolver.from_database(
        default_database.scoring.elec, torch_device
    )
    numpy.testing.assert_array_equal(
        resolver.get_partial_charges_for_block(water),
        numpy.zeros(3, dtype=numpy.float32),
    )
