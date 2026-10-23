from tmol._load_ext import _load_module

_mod = _load_module(
    __name__,
    __file__,
    "uaid_util.pybind.cc",
    "tmol.tests.score.common._uaid_util",
)
resolve_uaids = _mod.resolve_uaids

__all__ = ["resolve_uaids"]
