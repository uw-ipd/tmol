"""Score functions, energy terms, and standard weight sets."""

from __future__ import annotations

import toolz.functoolz
import torch
import os

from tmol.database import ParameterDatabase
from tmol.utility import resolve_device

from typing import Optional, TYPE_CHECKING

# Import in topological order: dependencies before dependents.
from . import _score_types  # noqa: F401
from . import _bonded_atom  # noqa: F401
from . import _chemical_database  # noqa: F401
from . import _energy_term  # noqa: F401
from . import _atom_type_dependent_term  # noqa: F401
from . import _bond_dependent_term  # noqa: F401
from . import _fragment_interactions  # noqa: F401
from ._score_function import ScoreFunction  # noqa: F401

if TYPE_CHECKING:
    pass


def _non_memoized_beta2016(
    device: torch.device, param_db: Optional[ParameterDatabase] = None
) -> "ScoreFunction":
    """Build a beta_nov2016 score function without memoization."""
    if param_db is None:
        param_db = ParameterDatabase.get_default()

    _weights_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "database",
        "score_functions",
        "beta2016.sfxn",
    )
    return ScoreFunction.from_sfxn_file(_weights_path, param_db, device)


@toolz.functoolz.memoize
def _memoized_beta2016(device: torch.device) -> "ScoreFunction":
    """Build and cache a score function keyed by device."""
    return _non_memoized_beta2016(device, None)


def beta2016_score_function(
    device: torch.device, param_db: Optional[ParameterDatabase] = None
) -> "ScoreFunction":
    """Return a ScoreFunction implementing the beta_nov2016_cart score function
    of Rosetta3.

    Note that in Rosetta3, beta_nov2016 and beta_nov2016_cart are identical
    except for the inclusion of the bond-length, bond-angle, and bond-torsion
    terms implemented by the CartBonded energy term, and the exclusion
    of the ProClose energy term (which is not implemented in tmol).

    Args:
        device: Target torch device.
        param_db: Optional parameter database. If omitted, uses the process
            default parameter database and a memoized score function.

    Returns:
        Configured `ScoreFunction`.

    When `param_db` is provided, this creates a fresh score function
    (no memoization — caller owns database lifecycle).

    See:
    https://pubs.acs.org/doi/10.1021/acs.jctc.6b0081 and
    https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00125
    """
    device = resolve_device(device)
    if param_db is not None:
        return _non_memoized_beta2016(device, param_db)
    return _memoized_beta2016(device)


__all__ = [
    "BackboneTorsionEnergyTerm",
    "CartBondedEnergyTerm",
    "ConstraintEnergyTerm",
    "DisulfideEnergyTerm",
    "DunbrackEnergyTerm",
    "ElecEnergyTerm",
    "GenBondedEnergyTerm",
    "HBondEnergyTerm",
    "LJLKEnergyTerm",
    "LKBallEnergyTerm",
    "NaTorsionEnergyTerm",
    "RefEnergyTerm",
    "ScoreTermFactory",
    "constrain_all_ca",
    "create_mainchain_coordinate_constraints",
]


# API names defined in modules this package does not otherwise import; they
# load on first access so importing the package stays cheap and acyclic.
_LAZY_ATTRS = {
    "BackboneTorsionEnergyTerm": "tmol.score.backbone_torsion._bb_torsion_energy_term",
    "CartBondedEnergyTerm": "tmol.score.cartbonded._cartbonded_energy_term",
    "ConstraintEnergyTerm": "tmol.score.constraint._constraint_energy_term",
    "DisulfideEnergyTerm": "tmol.score.disulfide._disulfide_energy_term",
    "DunbrackEnergyTerm": "tmol.score.dunbrack._dunbrack_energy_term",
    "ElecEnergyTerm": "tmol.score.elec._elec_energy_term",
    "GenBondedEnergyTerm": "tmol.score.genbonded._genbonded_energy_term",
    "HBondEnergyTerm": "tmol.score.hbond._hbond_energy_term",
    "LJLKEnergyTerm": "tmol.score.ljlk._ljlk_energy_term",
    "LKBallEnergyTerm": "tmol.score.lk_ball._lk_ball_energy_term",
    "NaTorsionEnergyTerm": "tmol.score.na_torsion._na_torsion_energy_term",
    "RefEnergyTerm": "tmol.score.ref._ref_energy_term",
    "ScoreTermFactory": "tmol.score.terms._score_term_factory",
    "constrain_all_ca": "tmol.score.constraint._utility",
    "create_mainchain_coordinate_constraints": "tmol.score.constraint._utility",
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
