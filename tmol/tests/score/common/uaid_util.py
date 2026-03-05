from tmol._load_ext import ensure_compiled_or_jit

if ensure_compiled_or_jit():
    from tmol.utility.cpp_extension import load, modulename, relpaths

    _mod = load(
        modulename(__name__),
        relpaths(__file__, ["uaid_util.pybind.cc"]),
        is_python_module=True,
    )
    resolve_uaids = _mod.resolve_uaids
else:
    from tmol.tests.score.common._uaid_util import resolve_uaids

__all__ = ["resolve_uaids"]
