import torch

from tmol.database import ParameterDatabase


def test_get_default():
    param_db1 = ParameterDatabase.get_default()
    param_db2 = ParameterDatabase.get_default()
    assert param_db1 is param_db2


def test_create_stable_subset(default_database):
    rt_name_subset = ["ALA", "ASN", "ARG", "PHE"]
    variant_subset = ["nterm", "cterm"]
    subset = default_database.create_stable_subset(rt_name_subset, variant_subset)
    assert len(subset.chemical.residues) == 16


def test_create_stable_subset_error_handline(default_database):
    rt_name_subset = ["ALA", "ASN", "ARG", "FEE"]
    variant_subset = ["nterm", "cterm"]
    try:
        default_database.create_stable_subset(rt_name_subset, variant_subset)
        assert False
    except ValueError as e:
        gold_error = "ERROR: could not build the requested PachedChemcialDatabase subset because 'FEE' is not present in the original set"
        assert str(e) == gold_error
