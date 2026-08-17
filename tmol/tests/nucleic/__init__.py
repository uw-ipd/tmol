_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "DNA_NAME3S": ("test_na_biotite_round_trip", "DNA_NAME3S"),
    "test_dna_five_prime_phosphate_is_not_required": (
        "test_na_biotite_round_trip",
        "test_dna_five_prime_phosphate_is_not_required",
    ),
    "test_dna_biotite_round_trip": (
        "test_na_biotite_round_trip",
        "test_dna_biotite_round_trip",
    ),
    "test_dna_biotite_coords_preserved": (
        "test_na_biotite_round_trip",
        "test_dna_biotite_coords_preserved",
    ),
    "test_dna_biotite_scores_are_finite": (
        "test_na_biotite_round_trip",
        "test_dna_biotite_scores_are_finite",
    ),
    "test_dna_biotite_keeps_all_nucleotides": (
        "test_na_biotite_round_trip",
        "test_dna_biotite_keeps_all_nucleotides",
    ),
    "RNA_NAME3S": ("test_na_round_trip", "RNA_NAME3S"),
    "RNA_BASE_NAMES": ("test_na_round_trip", "RNA_BASE_NAMES"),
    "NA_MAINCHAIN": ("test_na_round_trip", "NA_MAINCHAIN"),
    "test_na_restypes_in_canonical_ordering": (
        "test_na_round_trip",
        "test_na_restypes_in_canonical_ordering",
    ),
    "test_na_pose_stack_round_trip": (
        "test_na_round_trip",
        "test_na_pose_stack_round_trip",
    ),
    "test_na_pose_stack_coords_are_all_resolved": (
        "test_na_round_trip",
        "test_na_pose_stack_coords_are_all_resolved",
    ),
    "test_dna_termini_block_types": (
        "test_na_round_trip",
        "test_dna_termini_block_types",
    ),
    "test_rna_termini_block_types": (
        "test_na_round_trip",
        "test_rna_termini_block_types",
    ),
    "test_rna_ribose_is_not_deoxy": (
        "test_na_round_trip",
        "test_rna_ribose_is_not_deoxy",
    ),
    "test_protein_dna_chain_composition": (
        "test_na_round_trip",
        "test_protein_dna_chain_composition",
    ),
    "NA_ACTIVE_TERMS": ("test_na_scoring", "NA_ACTIVE_TERMS"),
    "NA_PLANARITY_TERMS": ("test_na_scoring", "NA_PLANARITY_TERMS"),
    "NA_INACTIVE_TERMS": ("test_na_scoring", "NA_INACTIVE_TERMS"),
    "test_beta2016_scores_na_are_finite": (
        "test_na_scoring",
        "test_beta2016_scores_na_are_finite",
    ),
    "test_beta2016_parameterized_terms_see_dna": (
        "test_na_scoring",
        "test_beta2016_parameterized_terms_see_dna",
    ),
    "test_beta2016_unparameterized_terms_are_zero_for_dna": (
        "test_na_scoring",
        "test_beta2016_unparameterized_terms_are_zero_for_dna",
    ),
    "test_beta2016_protein_dna_is_more_than_parts": (
        "test_na_scoring",
        "test_beta2016_protein_dna_is_more_than_parts",
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
