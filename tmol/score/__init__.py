"""Score functions, energy terms, and standard weight sets."""

from __future__ import annotations

import toolz.functoolz
import torch
import os

from tmol.database import ParameterDatabase
from tmol.utility import resolve_device

from typing import Optional, TYPE_CHECKING

# Import in topological order: dependencies before dependents.
from ._score_function import ScoreFunction  # noqa: F401
from .backbone_torsion._bb_torsion_energy_term import BackboneTorsionEnergyTerm
from .cartbonded._cartbonded_energy_term import CartBondedEnergyTerm
from .constraint._utility import create_mainchain_coordinate_constraints
from .constraint._constraint_energy_term import ConstraintEnergyTerm
from .disulfide._disulfide_energy_term import DisulfideEnergyTerm
from .dunbrack._dunbrack_energy_term import DunbrackEnergyTerm
from .elec._elec_energy_term import ElecEnergyTerm
from .genbonded._genbonded_energy_term import GenBondedEnergyTerm
from .hbond._hbond_energy_term import HBondEnergyTerm
from .ljlk._ljlk_energy_term import LJLKEnergyTerm
from .lk_ball._lk_ball_energy_term import LKBallEnergyTerm
from .na_torsion._na_torsion_energy_term import NaTorsionEnergyTerm
from .ref._ref_energy_term import RefEnergyTerm
from .terms._score_term_factory import ScoreTermFactory

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
    "ScoreFunction",
    "beta2016_score_function",
    "BackboneTorsionEnergyTerm",
    "CartBondedEnergyTerm",
    "create_mainchain_coordinate_constraints",
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
]
