_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "convert_float64": ("convert_float64", "convert_float64"),
    "interpolate": ("cubic_hermite_polynomial", "interpolate"),
    "interpolate_dt": ("cubic_hermite_polynomial", "interpolate_dt"),
    "interpolate_dx": ("cubic_hermite_polynomial", "interpolate_dx"),
    "interpolate_t": ("cubic_hermite_polynomial", "interpolate_t"),
    "interpolate_to_zero": ("cubic_hermite_polynomial", "interpolate_to_zero"),
    "interpolate_to_zero_V_dV": (
        "cubic_hermite_polynomial",
        "interpolate_to_zero_V_dV",
    ),
    "interpolate_to_zero_dt": ("cubic_hermite_polynomial", "interpolate_to_zero_dt"),
    "interpolate_to_zero_dx": ("cubic_hermite_polynomial", "interpolate_to_zero_dx"),
    "interpolate_to_zero_t": ("cubic_hermite_polynomial", "interpolate_to_zero_t"),
    "make_hashtable_keys_values": ("hash_util", "make_hashtable_keys_values"),
    "hash_fun": ("hash_util", "hash_fun"),
    "add_to_hashtable": ("hash_util", "add_to_hashtable"),
    "TermScoringModule": ("scoring_module", "TermScoringModule"),
    "TermPoseScoringModule": ("scoring_module", "TermPoseScoringModule"),
    "TermWholePoseScoringModule": ("scoring_module", "TermWholePoseScoringModule"),
    "TermBlockPairScoringModule": ("scoring_module", "TermBlockPairScoringModule"),
    "TermRotamerScoringModule": ("scoring_module", "TermRotamerScoringModule"),
    "condense_numpy_inds": ("stack_condense", "condense_numpy_inds"),
    "condense_torch_inds": ("stack_condense", "condense_torch_inds"),
    "take_values_w_sentineled_index": (
        "stack_condense",
        "take_values_w_sentineled_index",
    ),
    "take_values_w_sentineled_index_and_dest": (
        "stack_condense",
        "take_values_w_sentineled_index_and_dest",
    ),
    "take_values_w_sentineled_dest": (
        "stack_condense",
        "take_values_w_sentineled_dest",
    ),
    "condense_subset": ("stack_condense", "condense_subset"),
    "take_condensed_3d_subset": ("stack_condense", "take_condensed_3d_subset"),
    "tile_subset_indices": ("stack_condense", "tile_subset_indices"),
    "arg_tile_subset_indices": ("stack_condense", "arg_tile_subset_indices"),
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
