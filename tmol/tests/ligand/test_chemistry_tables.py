from tmol.database import ParameterDatabase
from tmol.ligand.chemistry_tables import (
    get_hbond_properties,
    get_polar_classes,
    get_sp2_atom_types,
)


def test_hbond_properties_derived_from_chemical_db():
    param_db = ParameterDatabase.get_default()
    atom_types_by_name = {at.name: at for at in param_db.chemical.atom_types}
    hbond_props = get_hbond_properties()

    assert "Ohx" in hbond_props
    assert hbond_props["Ohx"]["is_acceptor"] == atom_types_by_name["Ohx"].is_acceptor
    assert hbond_props["Ohx"]["is_donor"] == atom_types_by_name["Ohx"].is_donor
    assert "acceptor_hybridization" in hbond_props["Ohx"]


def test_polar_and_sp2_classes_come_from_db_tables():
    polar_classes = get_polar_classes()
    sp2_atom_types = get_sp2_atom_types()

    assert "PG3" in polar_classes
    assert "CD" in sp2_atom_types
    assert "Nim" in sp2_atom_types
