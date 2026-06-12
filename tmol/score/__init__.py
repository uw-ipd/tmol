from __future__ import annotations

import toolz.functoolz
import torch
import os

from tmol.database import ParameterDatabase

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .score_function import ScoreFunction


def _non_memoized_beta2016(
    device: torch.device, param_db: Optional[ParameterDatabase] = None
) -> ScoreFunction:
    """Build a beta_nov2016 score function without memoization."""
    from tmol.database import ParameterDatabase
    from .score_function import ScoreFunction

    if param_db is None:
        param_db = ParameterDatabase.get_default()

    _weights_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "database",
        "score_functions",
        "beta2016.yaml",
    )
    return ScoreFunction.from_weights_file(_weights_path, param_db, device)


@toolz.functoolz.memoize
def _memoized_beta2016(device: torch.device) -> ScoreFunction:
    """Build and cache a score function keyed by device."""
    return _non_memoized_beta2016(device, None)


def beta2016_score_function(
    device: torch.device, param_db: Optional[ParameterDatabase] = None
) -> ScoreFunction:
    """Return a ScoreFunction implementing the beta_nov2016 score function
    of Rosetta3.

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
    if param_db is not None:
        return _non_memoized_beta2016(device, param_db)
    return _memoized_beta2016(device)
