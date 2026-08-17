from .compiled import gen_pose_leaf_atoms, resolve_his_taut  # noqa: F401

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "gen_pose_leaf_atoms": ("compiled", "gen_pose_leaf_atoms"),
    "resolve_his_taut": ("compiled", "resolve_his_taut"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib

        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
