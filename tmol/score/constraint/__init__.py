from .constraint_energy_term import HiddenPrints, ConstraintEnergyTerm  # noqa: F401
from .utility import (  # noqa: F401
    constrain_all_ca,
    MCAtomIndices,
    create_mainchain_coordinate_constraints,
)

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "ConstraintEnergyTerm": ("constraint_energy_term", "ConstraintEnergyTerm"),
    "create_mainchain_coordinate_constraints": (
        "utility",
        "create_mainchain_coordinate_constraints",
    ),
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
