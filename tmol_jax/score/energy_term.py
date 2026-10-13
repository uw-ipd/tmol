"""Interface for a term's topology setup and differentiable energy evaluation."""

from abc import ABC, abstractmethod


class EnergyTerm(ABC):
    """Terms return unweighted channels; ScoreFunction applies their weights."""

    def __init__(self, param_db):
        self.param_db = param_db

    @abstractmethod
    def score_types_provided(self):
        """ScoreType channels implemented by this term."""
        raise NotImplementedError

    def setup_for_pose_stack(self, pose_stack, pbt=None):
        """Return a pytree of coordinate-independent arrays for this topology."""
        return {}

    @abstractmethod
    def __call__(self, pose_stack, term_state, coords_override=None):
        """Return {ScoreType: energy[poses]} without mutating state.

        Use coords_override when supplied; otherwise use pose_stack.coords.
        This evaluation must support jax.jit, differentiation and batching.
        """
        raise NotImplementedError
