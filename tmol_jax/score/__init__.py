"""JAX scoring and the canonical protein beta2016 score preset."""

import yaml

from .energy_term import EnergyTerm
from .score_function import ScoreFunction
from .score_types import ScoreType


def beta2016_score_function(param_db=None):
    """Construct a fresh protein score configuration using shared TMol weights.

    Nucleic acids, metals and noncanonical residues are not yet supported.
    """
    from tmol_jax.database import ParameterDatabase, default_directory

    if param_db is None:
        param_db = ParameterDatabase.get_default()
    path = default_directory() / "score_functions/beta2016.sfxn"
    with path.open() as source:
        weights = yaml.safe_load(source)["weights"]
    score = ScoreFunction(param_db)
    for name, weight in weights.items():
        if name not in {"metal_coordination", "na_torsion", "na_torsion_well"}:
            score.set_weight(ScoreType[name], weight)
    return score


__all__ = [
    "ScoreFunction",
    "ScoreType",
    "EnergyTerm",
    "beta2016_score_function",
]
