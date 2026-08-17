_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "test_identify_sidechain_atoms_from_roots": (
        "test_bfs_sidechain",
        "test_identify_sidechain_atoms_from_roots",
    ),
    "test_chi_atom_table_orders_double_digit_chis_numerically": (
        "test_build_rotamers",
        "test_chi_atom_table_orders_double_digit_chis_numerically",
    ),
    "test_build_rotamers_smoke": ("test_build_rotamers", "test_build_rotamers_smoke"),
    "test_construct_scans_for_rotamers": (
        "test_build_rotamers",
        "test_construct_scans_for_rotamers",
    ),
    "test_construct_scans_for_rotamers2": (
        "test_build_rotamers",
        "test_construct_scans_for_rotamers2",
    ),
    "test_measure_pose_dofs": ("test_build_rotamers", "test_measure_pose_dofs"),
    "test_inv_kin_rotamers": ("test_build_rotamers", "test_inv_kin_rotamers"),
    "test_construct_kinforest_for_rotamers": (
        "test_build_rotamers",
        "test_construct_kinforest_for_rotamers",
    ),
    "test_construct_kinforest_for_rotamers2": (
        "test_build_rotamers",
        "test_construct_kinforest_for_rotamers2",
    ),
    "test_measure_original_dofs": ("test_build_rotamers", "test_measure_original_dofs"),
    "test_measure_original_dofs2": (
        "test_build_rotamers",
        "test_measure_original_dofs2",
    ),
    "test_create_dof_inds_to_copy_from_orig_to_rotamers": (
        "test_build_rotamers",
        "test_create_dof_inds_to_copy_from_orig_to_rotamers",
    ),
    "test_create_dof_inds_to_copy_from_orig_to_rotamers2": (
        "test_build_rotamers",
        "test_create_dof_inds_to_copy_from_orig_to_rotamers2",
    ),
    "test_write_rotamers_pdb": ("test_build_rotamers", "test_write_rotamers_pdb"),
    "test_build_some_rotamers": ("test_build_rotamers", "test_build_some_rotamers"),
    "test_build_lots_of_rotamers": (
        "test_build_rotamers",
        "test_build_lots_of_rotamers",
    ),
    "test_score_lots_of_rotamers": (
        "test_build_rotamers",
        "test_score_lots_of_rotamers",
    ),
    "test_create_dofs_for_many_rotamers": (
        "test_build_rotamers",
        "test_create_dofs_for_many_rotamers",
    ),
    "test_new_rotamer_building_logic1": (
        "test_build_rotamers",
        "test_new_rotamer_building_logic1",
    ),
    "test_new_rotamer_building_logic2": (
        "test_build_rotamers",
        "test_new_rotamer_building_logic2",
    ),
    "test_new_rotamer_building_logic3": (
        "test_build_rotamers",
        "test_new_rotamer_building_logic3",
    ),
    "test_chi_sampler_smoke": ("test_fixed_aa_chi_sampler", "test_chi_sampler_smoke"),
    "test_annotate_residue_type_smoke": (
        "test_include_current_sampler",
        "test_annotate_residue_type_smoke",
    ),
    "test_annotate_packed_block_types_smoke": (
        "test_include_current_sampler",
        "test_annotate_packed_block_types_smoke",
    ),
    "test_include_current_sampler_smoke": (
        "test_include_current_sampler",
        "test_include_current_sampler_smoke",
    ),
    "test_create_non_sidechain_fingerprint": (
        "test_mainchain_fingerprint",
        "test_create_non_sidechain_fingerprint",
    ),
    "test_create_non_sc_fingerprint_smoke": (
        "test_mainchain_fingerprint",
        "test_create_non_sc_fingerprint_smoke",
    ),
    "test_annotate_rt_w_mainchain_fingerprint": (
        "test_mainchain_fingerprint",
        "test_annotate_rt_w_mainchain_fingerprint",
    ),
    "test_merge_fingerprints": (
        "test_mainchain_fingerprint",
        "test_merge_fingerprints",
    ),
    "test_na_sampler_builds_for_nucleotides_only": (
        "test_na_chi_sampler",
        "test_na_sampler_builds_for_nucleotides_only",
    ),
    "test_na_sampler_chi_sit_at_scored_minima": (
        "test_na_chi_sampler",
        "test_na_sampler_chi_sit_at_scored_minima",
    ),
    "test_na_sampler_syn_is_rna_only": (
        "test_na_chi_sampler",
        "test_na_sampler_syn_is_rna_only",
    ),
    "test_na_sampler_expands_proton_chis": (
        "test_na_chi_sampler",
        "test_na_sampler_expands_proton_chis",
    ),
    "test_na_sampler_chi_level_widens_the_rotamer_set": (
        "test_na_chi_sampler",
        "test_na_sampler_chi_level_widens_the_rotamer_set",
    ),
    "test_na_build_rotamers": ("test_na_chi_sampler", "test_na_build_rotamers"),
    "test_opth_builds_cartesian_product_for_multiple_proton_chis": (
        "test_opth_sampler",
        "test_opth_builds_cartesian_product_for_multiple_proton_chis",
    ),
    "test_optH_rotamer_sampler_flipNHQ": (
        "test_opth_sampler",
        "test_optH_rotamer_sampler_flipNHQ",
    ),
    "test_optH_rotamer_sampler_no_flipNHQ": (
        "test_opth_sampler",
        "test_optH_rotamer_sampler_no_flipNHQ",
    ),
    "test_annotate_restypes": ("test_single_residue_kintree", "test_annotate_restypes"),
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
