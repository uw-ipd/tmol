# init!

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "bfs_sidechain_atoms_jit": ("bfs_sidechain", "bfs_sidechain_atoms_jit"),
    "bfs_sidechain_atoms": ("bfs_sidechain", "bfs_sidechain_atoms"),
    "correct_phi_c_for_jump_parents": (
        "build_rotamers",
        "correct_phi_c_for_jump_parents",
    ),
    "exc_cumsum_from_inc_cumsum": ("build_rotamers", "exc_cumsum_from_inc_cumsum"),
    "annotate_restype": ("build_rotamers", "annotate_restype"),
    "annotate_packed_block_types": ("build_rotamers", "annotate_packed_block_types"),
    "annotate_everything": ("build_rotamers", "annotate_everything"),
    "update_nodes": ("build_rotamers", "update_nodes"),
    "update_scan_starts": ("build_rotamers", "update_scan_starts"),
    "construct_scans_for_conformers": (
        "build_rotamers",
        "construct_scans_for_conformers",
    ),
    "load_from_rotamers": ("build_rotamers", "load_from_rotamers"),
    "load_from_rotamers_w_offsets": ("build_rotamers", "load_from_rotamers_w_offsets"),
    "load_rotamer_parents": ("build_rotamers", "load_rotamer_parents"),
    "construct_kinforest_for_conformers": (
        "build_rotamers",
        "construct_kinforest_for_conformers",
    ),
    "measure_dofs_from_orig_coords": (
        "build_rotamers",
        "measure_dofs_from_orig_coords",
    ),
    "measure_pose_dofs": ("build_rotamers", "measure_pose_dofs"),
    "merge_conformer_samples": ("build_rotamers", "merge_conformer_samples"),
    "calculate_rotamer_coords": ("build_rotamers", "calculate_rotamer_coords"),
    "get_rotamer_origin_data": ("build_rotamers", "get_rotamer_origin_data"),
    "build_rotamers": ("build_rotamers", "build_rotamers"),
    "ChiSampler": ("chi_sampler", "ChiSampler"),
    "copy_dofs_from_orig_to_rotamers_for_sampler": (
        "chi_sampler",
        "copy_dofs_from_orig_to_rotamers_for_sampler",
    ),
    "create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler": (
        "chi_sampler",
        "create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler",
    ),
    "assign_chi_dofs_from_samples": ("chi_sampler", "assign_chi_dofs_from_samples"),
    "ConformerSampler": ("conformer_sampler", "ConformerSampler"),
    "FallbackSampler": ("fallback_sampler", "FallbackSampler"),
    "FixedAAChiSampler": ("fixed_aa_chi_sampler", "FixedAAChiSampler"),
    "IncludeCurrentSampler": ("include_current_sampler", "IncludeCurrentSampler"),
    "create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler": (
        "include_current_sampler",
        "create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler",
    ),
    "AtomFingerprint": ("mainchain_fingerprint", "AtomFingerprint"),
    "MCFingerprint": ("mainchain_fingerprint", "MCFingerprint"),
    "MCFingerprints": ("mainchain_fingerprint", "MCFingerprints"),
    "create_non_sidechain_fingerprint": (
        "mainchain_fingerprint",
        "create_non_sidechain_fingerprint",
    ),
    "create_mainchain_fingerprint": (
        "mainchain_fingerprint",
        "create_mainchain_fingerprint",
    ),
    "annotate_residue_type_with_sampler_fingerprints": (
        "mainchain_fingerprint",
        "annotate_residue_type_with_sampler_fingerprints",
    ),
    "find_max_length_fp_among_res_samplers": (
        "mainchain_fingerprint",
        "find_max_length_fp_among_res_samplers",
    ),
    "find_unique_fingerprints": ("mainchain_fingerprint", "find_unique_fingerprints"),
    "CHI_STEPS": ("na_chi_sampler", "CHI_STEPS"),
    "MAX_SYN_WELL": ("na_chi_sampler", "MAX_SYN_WELL"),
    "NA_PROTON_CHI_ROOT": ("na_chi_sampler", "NA_PROTON_CHI_ROOT"),
    "na_proton_chi_roots": ("na_chi_sampler", "na_proton_chi_roots"),
    "NaChiRotamerSampler": ("na_chi_sampler", "NaChiRotamerSampler"),
    "OptHSamplerRTCache": ("opth_sampler", "OptHSamplerRTCache"),
    "OptHSamplerPackedBlockTypeCache": (
        "opth_sampler",
        "OptHSamplerPackedBlockTypeCache",
    ),
    "OptHSampler": ("opth_sampler", "OptHSampler"),
    "RotamerSet": ("rotamer_set", "RotamerSet"),
    "RotamerKintree": ("single_residue_kinforest", "RotamerKintree"),
    "PackedRotamerKintree": ("single_residue_kinforest", "PackedRotamerKintree"),
    "construct_single_residue_kinforest": (
        "single_residue_kinforest",
        "construct_single_residue_kinforest",
    ),
    "coalesce_single_residue_kinforests": (
        "single_residue_kinforest",
        "coalesce_single_residue_kinforests",
    ),
    "_build_chi4_atom_table": ("build_rotamers", "_build_chi4_atom_table"),
    "_build_chi_phi_c_corrections": ("build_rotamers", "_build_chi_phi_c_corrections"),
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
