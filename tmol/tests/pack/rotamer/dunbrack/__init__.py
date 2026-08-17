import pytest

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "get_compiled": ("test_dunbrack_chi_sampler", "get_compiled"),
    "test_annotate_residue_type": (
        "test_dunbrack_chi_sampler",
        "test_annotate_residue_type",
    ),
    "test_annotate_packed_block_types": (
        "test_dunbrack_chi_sampler",
        "test_annotate_packed_block_types",
    ),
    "test_determine_n_possible_rots": (
        "test_dunbrack_chi_sampler",
        "test_determine_n_possible_rots",
    ),
    "test_fill_in_brt_for_possrots": (
        "test_dunbrack_chi_sampler",
        "test_fill_in_brt_for_possrots",
    ),
    "test_interpolate_probabilities_for_possible_rotamers": (
        "test_dunbrack_chi_sampler",
        "test_interpolate_probabilities_for_possible_rotamers",
    ),
    "test_determine_n_base_rotamers_to_build_1": (
        "test_dunbrack_chi_sampler",
        "test_determine_n_base_rotamers_to_build_1",
    ),
    "test_determine_n_base_rotamers_to_build_2": (
        "test_dunbrack_chi_sampler",
        "test_determine_n_base_rotamers_to_build_2",
    ),
    "test_count_expanded_rotamers": (
        "test_dunbrack_chi_sampler",
        "test_count_expanded_rotamers",
    ),
    "test_map_from_rotamer_index_to_brt": (
        "test_dunbrack_chi_sampler",
        "test_map_from_rotamer_index_to_brt",
    ),
    "test_sample_chi_for_rotamers": (
        "test_dunbrack_chi_sampler",
        "test_sample_chi_for_rotamers",
    ),
    "test_package_samples_for_output": (
        "test_dunbrack_chi_sampler",
        "test_package_samples_for_output",
    ),
    "test_chi_sampler_smoke": ("test_dunbrack_chi_sampler", "test_chi_sampler_smoke"),
    "test_chi_sampler_build_lots_of_rotamers": (
        "test_dunbrack_chi_sampler",
        "test_chi_sampler_build_lots_of_rotamers",
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


@pytest.fixture()
def dun_sampler(default_database, torch_device):
    from tmol.pack.rotamer.dunbrack import (
        create_dunbrack_sampler_from_database,
    )

    return create_dunbrack_sampler_from_database(default_database, torch_device)
