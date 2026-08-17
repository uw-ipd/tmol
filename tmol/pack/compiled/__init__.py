from .compiled import (  # noqa: F401
    pack_anneal,
    validate_energies,
    build_interaction_graph,
)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "build_interaction_graph": ("compiled", "build_interaction_graph"),
    "pack_anneal": ("compiled", "pack_anneal"),
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
