"""Side-chain packing tasks, energy tables, and annealing."""

from . import _datatypes  # noqa: F401
from ._packer_task import SetPackerTask  # noqa: F401
from . import _simulated_annealing  # noqa: F401
from . import _impose_rotamers  # noqa: F401
from . import _pack_rotamers  # noqa: F401
from . import _build_missing_sidechains  # noqa: F401
from . import _mutate  # noqa: F401

__all__ = [
    "ChiSampler",
    "ConjugatedSCSampler",
    "DunbrackChiSampler",
    "FallbackSampler",
    "FixedAAChiSampler",
    "IncludeCurrentSampler",
    "NaChiRotamerSampler",
    "OptHSampler",
    "RotamerSet",
    "SetPackerTask",
    "SidechainSampler",
    "build_rotamers",
    "create_dunbrack_sampler_from_database",
]


# API names defined in modules this package does not otherwise import; they
# load on first access so importing the package stays cheap and acyclic.
_LAZY_ATTRS = {
    "ChiSampler": "tmol.pack.rotamer._chi_sampler",
    "ConjugatedSCSampler": "tmol.pack.rotamer._sidechain_sampler",
    "DunbrackChiSampler": "tmol.pack.rotamer.dunbrack._dunbrack_chi_sampler",
    "FallbackSampler": "tmol.pack.rotamer._fallback_sampler",
    "FixedAAChiSampler": "tmol.pack.rotamer._fixed_aa_chi_sampler",
    "IncludeCurrentSampler": "tmol.pack.rotamer._include_current_sampler",
    "NaChiRotamerSampler": "tmol.pack.rotamer._na_chi_sampler",
    "OptHSampler": "tmol.pack.rotamer._opth_sampler",
    "RotamerSet": "tmol.pack.rotamer._rotamer_set",
    "SidechainSampler": "tmol.pack.rotamer._sidechain_sampler",
    "build_rotamers": "tmol.pack.rotamer._build_rotamers",
    "create_dunbrack_sampler_from_database": "tmol.pack.rotamer.dunbrack._dunbrack_chi_sampler",
}


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
