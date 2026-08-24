from __future__ import annotations

import toolz.functoolz
import torch
import os

from tmol.database import ParameterDatabase
from tmol.utility import resolve_device

from typing import Optional, TYPE_CHECKING

# Import in topological order: dependencies before dependents.
from ._score_types import ScoreType  # noqa: F401
from ._bonded_atom import IndexedBonds  # noqa: F401
from ._chemical_database import (  # noqa: F401
    AcceptorHybridization,
    AtomTypeParamResolver,
    AtomTypeParams,
)  # noqa: F401
from ._energy_term import EnergyTerm  # noqa: F401
from ._atom_type_dependent_term import AtomTypeDependentTerm  # noqa: F401
from ._bond_dependent_term import BondDependentTerm  # noqa: F401
from ._score_function import (  # noqa: F401
    BlockPairScoringModule,
    RotamerScoringModule,
    SFXN_FORMAT_VERSION,
    ScoreFunction,
    WholePoseScoringModule,
)

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
