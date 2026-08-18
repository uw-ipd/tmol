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
from ._score_utils import (  # noqa: F401
    FragmentInteractionScores,
    build_sidechain_coord_mask,
    build_coord_mask_for_mask_and_nearby_blocks,
    build_coord_mask_for_mask_and_interacting_atoms,
    compute_block_centroids_and_furthest_dist,
    calculate_block_pair_ddg,
    calculate_fragment_interactions,
    compute_block_adjacency,
    res_mask_to_coord_mask,
    residue_mask_from_chain,
)
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
    """Return a ScoreFunction implementing the beta_nov2016 score function of Rosetta3."""
    device = resolve_device(device)
    if param_db is not None:
        return _non_memoized_beta2016(device, param_db)
    return _memoized_beta2016(device)
