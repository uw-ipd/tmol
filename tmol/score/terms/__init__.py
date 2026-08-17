from os.path import dirname, basename, isfile, join
import glob

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "BackboneTorsionTermCreator": (
        "backbone_torsion_creator",
        "BackboneTorsionTermCreator",
    ),
    "CartBondedTermCreator": ("cartbonded_creator", "CartBondedTermCreator"),
    "ConstraintTermCreator": ("constraint_creator", "ConstraintTermCreator"),
    "DisulfideTermCreator": ("disulfide_creator", "DisulfideTermCreator"),
    "DunbrackTermCreator": ("dunbrack_creator", "DunbrackTermCreator"),
    "ElecTermCreator": ("elec_creator", "ElecTermCreator"),
    "GenBondedTermCreator": ("genbonded_creator", "GenBondedTermCreator"),
    "HBondTermCreator": ("hbond_creator", "HBondTermCreator"),
    "LJLKTermCreator": ("ljlk_creator", "LJLKTermCreator"),
    "LKBallTermCreator": ("lk_ball_creator", "LKBallTermCreator"),
    "NaTorsionTermCreator": ("na_torsion_creator", "NaTorsionTermCreator"),
    "RefTermCreator": ("ref_creator", "RefTermCreator"),
    "ScoreTermFactory": ("score_term_factory", "ScoreTermFactory"),
    "TermCreator": ("term_creator", "TermCreator"),
    "score_term_creator": ("term_creator", "score_term_creator"),
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


modules = glob.glob(join(dirname(__file__), "*.py"))
exclude = [join(dirname(__file__), f) for f in ["score_type_factory.py", "__init__.py"]]

__all__ = [
    basename(f)[:-3]
    for f in modules
    if isfile(f) and not f.endswith("__init__.py") and f not in exclude
]
