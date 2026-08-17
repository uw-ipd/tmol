from .fast_relax import (  # noqa: F401
    DEFAULT_RELAX_SCHEDULE,
    fast_relax,
    relax_pack_min_step,
    accept_best,
)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "_default_cart_min_fn": ("fast_relax", "_default_cart_min_fn"),
    "fast_relax": ("fast_relax", "fast_relax"),
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
