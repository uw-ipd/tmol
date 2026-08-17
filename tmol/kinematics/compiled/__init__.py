from .compiled_ops import forward_kin_op, inverse_kin

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "forward_only_op": ("compiled_ops", "forward_only_op"),
    "get_kfo_indices_for_atoms": ("compiled_ops", "get_kfo_indices_for_atoms"),
    "get_kfo_atom_parents": ("compiled_ops", "get_kfo_atom_parents"),
    "get_children": ("compiled_ops", "get_children"),
    "get_id_and_frame_xyz": ("compiled_ops", "get_id_and_frame_xyz"),
    "calculate_ff_edge_delays": ("compiled_ops", "calculate_ff_edge_delays"),
    "get_jump_atom_indices": ("compiled_ops", "get_jump_atom_indices"),
    "get_block_parent_connectivity_from_toposort": (
        "compiled_ops",
        "get_block_parent_connectivity_from_toposort",
    ),
    "get_kinforest_scans_from_stencils": (
        "compiled_ops",
        "get_kinforest_scans_from_stencils",
    ),
    "get_kinforest_scans_from_stencils2": (
        "compiled_ops",
        "get_kinforest_scans_from_stencils2",
    ),
    "minimizer_map_from_movemap": ("compiled_ops", "minimizer_map_from_movemap"),
    "forward_kin_op": ("compiled_ops", "forward_kin_op"),
    "inverse_kin": ("compiled_ops", "inverse_kin"),
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


__all__ = ["forward_kin_op", "inverse_kin"]
