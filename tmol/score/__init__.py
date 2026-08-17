from __future__ import annotations


import toolz.functoolz
import torch
import os

from tmol.database import ParameterDatabase
from tmol.utility import resolve_device

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .score_function import ScoreFunction


_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "AtomTypeDependentTerm": ("atom_type_dependent_term", "AtomTypeDependentTerm"),
    "BondDependentTerm": ("bond_dependent_term", "BondDependentTerm"),
    "IndexedBonds": ("bonded_atom", "IndexedBonds"),
    "AcceptorHybridization": ("chemical_database", "AcceptorHybridization"),
    "AtomTypeParams": ("chemical_database", "AtomTypeParams"),
    "AtomTypeParamResolver": ("chemical_database", "AtomTypeParamResolver"),
    "EnergyTerm": ("energy_term", "EnergyTerm"),
    "logger": ("score_function", "logger"),
    "SFXN_FORMAT_VERSION": ("score_function", "SFXN_FORMAT_VERSION"),
    "ScoreFunction": ("score_function", "ScoreFunction"),
    "WholePoseScoringModule": ("score_function", "WholePoseScoringModule"),
    "BlockPairScoringModule": ("score_function", "BlockPairScoringModule"),
    "RotamerScoringModule": ("score_function", "RotamerScoringModule"),
    "ScoreType": ("score_types", "ScoreType"),
    "FragmentInteractionScores": ("score_utils", "FragmentInteractionScores"),
    "calculate_fragment_interactions": (
        "score_utils",
        "calculate_fragment_interactions",
    ),
    "residue_mask_from_chain": ("score_utils", "residue_mask_from_chain"),
    "calculate_block_pair_ddg": ("score_utils", "calculate_block_pair_ddg"),
    "res_mask_to_coord_mask": ("score_utils", "res_mask_to_coord_mask"),
    "build_sidechain_coord_mask": ("score_utils", "build_sidechain_coord_mask"),
    "compute_block_centroids_and_furthest_dist": (
        "score_utils",
        "compute_block_centroids_and_furthest_dist",
    ),
    "build_coord_mask_for_mask_and_interacting_atoms": (
        "score_utils",
        "build_coord_mask_for_mask_and_interacting_atoms",
    ),
    "build_coord_mask_for_mask_and_nearby_blocks": (
        "score_utils",
        "build_coord_mask_for_mask_and_nearby_blocks",
    ),
    "compute_block_adjacency": ("score_utils", "compute_block_adjacency"),
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
        "beta2016.sfxn",
    )
    return ScoreFunction.from_sfxn_file(_weights_path, param_db, device)


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
    # resolved before the memo lookup so that 'cuda' and 'cuda:0' share one entry
    device = resolve_device(device)
    if param_db is not None:
        return _non_memoized_beta2016(device, param_db)
    return _memoized_beta2016(device)
