from tmol.score.terms._term_creator import TermCreator, _score_term_creator
from tmol.score._score_types import ScoreType
from tmol.database import ParameterDatabase
import torch


@_score_term_creator
class ConstraintTermCreator(TermCreator):
    """Create the pose-constraint energy term."""

    _score_types = [ScoreType.constraint]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.constraint._constraint_energy_term

        return tmol.score.constraint._constraint_energy_term.ConstraintEnergyTerm(
            param_db, device
        )

    @classmethod
    def score_types(cls):
        return cls._score_types
