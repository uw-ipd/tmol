import pytest

import tmol.chemical.restypes

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_patched_residue_construction_smoke": (
        "test_patch",
        "test_patched_residue_construction_smoke",
    ),
    "test_patched_residue_icoor_mapping": (
        "test_patch",
        "test_patched_residue_icoor_mapping",
    ),
    "test_patched_residue_ideal_coords": (
        "test_patch",
        "test_patched_residue_ideal_coords",
    ),
    "test_patched_pdb": ("test_patch", "test_patched_pdb"),
    "variant_from_yaml": ("test_patch", "variant_from_yaml"),
    "residues_from_yaml": ("test_patch", "residues_from_yaml"),
    "test_uncommon_patching_options": ("test_patch", "test_uncommon_patching_options"),
    "test_patch_error_checks": ("test_patch", "test_patch_error_checks"),
    "test_patch_validation_missing_fields": (
        "test_patch",
        "test_patch_validation_missing_fields",
    ),
    "test_patch_validation_remove_atoms_reference": (
        "test_patch",
        "test_patch_validation_remove_atoms_reference",
    ),
    "test_patch_validation_modify_atoms_reference": (
        "test_patch",
        "test_patch_validation_modify_atoms_reference",
    ),
    "test_patch_validation_illegal_add_alias": (
        "test_patch",
        "test_patch_validation_illegal_add_alias",
    ),
    "test_patch_validation_illegal_bond": (
        "test_patch",
        "test_patch_validation_illegal_bond",
    ),
    "test_patch_validation_illegal_icoor": (
        "test_patch",
        "test_patch_validation_illegal_icoor",
    ),
    "test_res_error_checks": ("test_patch", "test_res_error_checks"),
    "test_validate_restype_bad_conns": (
        "test_patch",
        "test_validate_restype_bad_conns",
    ),
    "test_validate_restype_bad_icoor": (
        "test_patch",
        "test_validate_restype_bad_icoor",
    ),
    "test_refined_residue_construction_smoke": (
        "test_residue",
        "test_refined_residue_construction_smoke",
    ),
    "test_refined_residue_icoor_mapping": (
        "test_residue",
        "test_refined_residue_icoor_mapping",
    ),
    "test_refined_residue_ideal_coords": (
        "test_residue",
        "test_refined_residue_ideal_coords",
    ),
    "test_refined_residue_ordered_torsions": (
        "test_residue",
        "test_refined_residue_ordered_torsions",
    ),
    "test_residue_type_set_construction": (
        "test_residue",
        "test_residue_type_set_construction",
    ),
    "test_residue_type_set_get_default": (
        "test_residue",
        "test_residue_type_set_get_default",
    ),
    "test_from_database_caching": ("test_residue", "test_from_database_caching"),
    "test_build_ideal_coords_smoke": ("test_residue", "test_build_ideal_coords_smoke"),
    "test_all_bonds_construction": ("test_residue", "test_all_bonds_construction"),
    "test_mc_sc_torsion_properties": ("test_residue", "test_mc_sc_torsion_properties"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        # Re-cache every name from this module so that Python's import
        # machinery (which sets globals()[mod_leaf] = MODULE as a side-effect)
        # does not overwrite previously resolved function/class references.
        for _n, (_m, _a) in _LAZY_ATTRS.items():
            if _m == mod_leaf:
                try:
                    globals()[_n] = getattr(mod, _a)
                except AttributeError:
                    pass
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@pytest.fixture
def default_restype_set():
    return tmol.chemical.restypes.ResidueTypeSet.get_default()


@pytest.fixture
def fresh_default_restype_set(default_database):
    """Fresh ResidueTypeSet constructed for each test"""
    return tmol.chemical.restypes.ResidueTypeSet.from_database(
        default_database.chemical
    )


@pytest.fixture()
def rts_disulfide_res(fresh_default_restype_set, disulfide_res):
    import attr

    rts = fresh_default_restype_set

    return [
        attr.evolve(
            res,
            residue_type=next(
                rt for rt in rts.residue_types if rt.name == res.residue_type.name
            ),
        )
        for res in disulfide_res
    ]
