_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "table_schema": ("hbond_param_import", "table_schema"),
    "RawParams": ("hbond_param_import", "RawParams"),
    "RosettaHBParams": ("hbond_param_import", "RosettaHBParams"),
    "basetype_for_dtype": ("hbond_param_import", "basetype_for_dtype"),
    "attrs_for_dtypes": ("hbond_param_import", "attrs_for_dtypes"),
    "HERE": ("na_torsion_param_import", "HERE"),
    "DNA_STRUCTURE_LIST": ("na_torsion_param_import", "DNA_STRUCTURE_LIST"),
    "RNA_STRUCTURE_LIST": ("na_torsion_param_import", "RNA_STRUCTURE_LIST"),
    "PURINE": ("na_torsion_param_import", "PURINE"),
    "PYRIMIDINE": ("na_torsion_param_import", "PYRIMIDINE"),
    "NA_RESNAMES": ("na_torsion_param_import", "NA_RESNAMES"),
    "BASE1": ("na_torsion_param_import", "BASE1"),
    "POLYMERS": ("na_torsion_param_import", "POLYMERS"),
    "BASE_ORDER": ("na_torsion_param_import", "BASE_ORDER"),
    "DEOXY_ONLY": ("na_torsion_param_import", "DEOXY_ONLY"),
    "RIBO_ONLY": ("na_torsion_param_import", "RIBO_ONLY"),
    "RING": ("na_torsion_param_import", "RING"),
    "SUGAR_TORSIONS": ("na_torsion_param_import", "SUGAR_TORSIONS"),
    "N_SUGAR": ("na_torsion_param_import", "N_SUGAR"),
    "N_PUCKER": ("na_torsion_param_import", "N_PUCKER"),
    "MIN_TORSIONS": ("na_torsion_param_import", "MIN_TORSIONS"),
    "PSEUDOCOUNT": ("na_torsion_param_import", "PSEUDOCOUNT"),
    "NORTH_PUCKERS": ("na_torsion_param_import", "NORTH_PUCKERS"),
    "MIN_CHI_TORSION": ("na_torsion_param_import", "MIN_CHI_TORSION"),
    "SDEV_BACKBONE": ("na_torsion_param_import", "SDEV_BACKBONE"),
    "SDEV_SUGAR": ("na_torsion_param_import", "SDEV_SUGAR"),
    "SDEV_CHI": ("na_torsion_param_import", "SDEV_CHI"),
    "WEIGHT_BB": ("na_torsion_param_import", "WEIGHT_BB"),
    "WEIGHT_CHI": ("na_torsion_param_import", "WEIGHT_CHI"),
    "WEIGHT_SUGAR": ("na_torsion_param_import", "WEIGHT_SUGAR"),
    "PUCKER_TEMPERATURE": ("na_torsion_param_import", "PUCKER_TEMPERATURE"),
    "BIN_BLEND_SDEV": ("na_torsion_param_import", "BIN_BLEND_SDEV"),
    "RCSB_SEARCH": ("na_torsion_param_import", "RCSB_SEARCH"),
    "RCSB_FILES": ("na_torsion_param_import", "RCSB_FILES"),
    "query_rcsb": ("na_torsion_param_import", "query_rcsb"),
    "fetch": ("na_torsion_param_import", "fetch"),
    "read_pdb": ("na_torsion_param_import", "read_pdb"),
    "read_cif": ("na_torsion_param_import", "read_cif"),
    "read_structure": ("na_torsion_param_import", "read_structure"),
    "dihedral": ("na_torsion_param_import", "dihedral"),
    "subtract_degree_angles": ("na_torsion_param_import", "subtract_degree_angles"),
    "triple_bin": ("na_torsion_param_import", "triple_bin"),
    "b1b2_bin": ("na_torsion_param_import", "b1b2_bin"),
    "sugar_pucker": ("na_torsion_param_import", "sugar_pucker"),
    "residue_torsions": ("na_torsion_param_import", "residue_torsions"),
    "polymer_of": ("na_torsion_param_import", "polymer_of"),
    "observations": ("na_torsion_param_import", "observations"),
    "circular_mean": ("na_torsion_param_import", "circular_mean"),
    "sugar_means": ("na_torsion_param_import", "sugar_means"),
    "backbone_means": ("na_torsion_param_import", "backbone_means"),
    "well_tables": ("na_torsion_param_import", "well_tables"),
    "observed_sdev": ("na_torsion_param_import", "observed_sdev"),
    "emit_yaml": ("na_torsion_param_import", "emit_yaml"),
    "read_rosetta_stats": ("na_torsion_param_import", "read_rosetta_stats"),
    "validate": ("na_torsion_param_import", "validate"),
    "build_tables": ("na_torsion_param_import", "build_tables"),
    "main": ("na_torsion_param_import", "main"),
    "rotamer_aliases": ("rewrite_dunbrack_binary", "rotamer_aliases"),
    "create_rotameric_data_for_aa": (
        "rewrite_dunbrack_binary",
        "create_rotameric_data_for_aa",
    ),
    "strip_comments": ("rewrite_dunbrack_binary", "strip_comments"),
    "create_rotameric_aa_dunbrack_library": (
        "rewrite_dunbrack_binary",
        "create_rotameric_aa_dunbrack_library",
    ),
    "create_semi_rotameric_aa_dunbrack_library": (
        "rewrite_dunbrack_binary",
        "create_semi_rotameric_aa_dunbrack_library",
    ),
    "create_dunbrack_rotamer_library": (
        "rewrite_dunbrack_binary",
        "create_dunbrack_rotamer_library",
    ),
    "parse_lines_as_ndarrays": (
        "rewrite_omega_bbdep_binary",
        "parse_lines_as_ndarrays",
    ),
    "parse_all_tables": ("rewrite_omega_bbdep_binary", "parse_all_tables"),
    "create_omega_db": ("rewrite_omega_bbdep_binary", "create_omega_db"),
    "parse_paa": ("rewrite_rama_binary", "parse_paa"),
    "create_rama_database": ("rewrite_rama_binary", "create_rama_database"),
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
