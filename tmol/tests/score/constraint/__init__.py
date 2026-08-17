_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "add_test_constraints_to_pose_stack": (
        "test_constraint_energy_term",
        "add_test_constraints_to_pose_stack",
    ),
    "check_fail_add_cross_pose_constraint": (
        "test_constraint_energy_term",
        "check_fail_add_cross_pose_constraint",
    ),
    "test_get_torsion_angle": ("test_constraint_energy_term", "test_get_torsion_angle"),
    "test_circularharmonic_periodic_values_and_derivatives": (
        "test_constraint_energy_term",
        "test_circularharmonic_periodic_values_and_derivatives",
    ),
    "test_circularharmonic_minus_pi_plus_pi_branch_equivalence": (
        "test_constraint_energy_term",
        "test_circularharmonic_minus_pi_plus_pi_branch_equivalence",
    ),
    "add_constraints_to_all_poses": (
        "test_constraint_energy_term",
        "add_constraints_to_all_poses",
    ),
    "modify_distances_and_check_constraints": (
        "test_constraint_energy_term",
        "modify_distances_and_check_constraints",
    ),
    "TestConstraintEnergyTerm": (
        "test_constraint_energy_term",
        "TestConstraintEnergyTerm",
    ),
    "test_create_coordinate_constraints": (
        "test_constraint_energy_term",
        "test_create_coordinate_constraints",
    ),
    "test_create_mainchain_coordinate_constraints": (
        "test_constraint_utilities",
        "test_create_mainchain_coordinate_constraints",
    ),
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
