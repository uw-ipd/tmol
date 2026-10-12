"""JAX scoring frontend for TMol; no PyTorch runtime dependency."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("tmol-jax")
except PackageNotFoundError:
    __version__ = "0+source"

from .io import pose_stack_from_pdb
from .optimization import run_cart_min
from .score import ScoreFunction, ScoreType, beta2016_score_function

__all__ = [
    "pose_stack_from_pdb",
    "run_cart_min",
    "ScoreFunction",
    "ScoreType",
    "beta2016_score_function",
]
