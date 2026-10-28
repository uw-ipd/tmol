"""Tests that ligand chemistry tables are derived from the chemical database."""

from tmol import ParameterDatabase
from tmol.ligand._chemistry_tables import _get_hbond_properties, _get_polar_classes


def test_hbond_properties_derived_from_chemical_db() -> None:
    """H-bond properties mirror the chemical database atom-type flags."""
    param_db = ParameterDatabase.get_default()
    atom_types_by_name = {at.name: at for at in param_db.chemical.atom_types}
    hbond_props = _get_hbond_properties()

    assert "Ohx" in hbond_props
    assert hbond_props["Ohx"]["is_acceptor"] == atom_types_by_name["Ohx"].is_acceptor
    assert hbond_props["Ohx"]["is_donor"] == atom_types_by_name["Ohx"].is_donor
    assert "acceptor_hybridization" in hbond_props["Ohx"]


def test_polar_classes_come_from_db_tables() -> None:
    """Polar atom-type classes are sourced from the database tables."""
    polar_classes = _get_polar_classes()

    assert "PG3" in polar_classes
