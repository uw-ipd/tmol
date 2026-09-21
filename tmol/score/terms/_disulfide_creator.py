from tmol.score.terms import TermCreator, score_term_creator
from tmol.score import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class DisulfideTermCreator(TermCreator):
    """Create the disulfide-geometry energy term."""

    _score_types = [ScoreType.disulfide]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.disulfide._disulfide_energy_term

        return tmol.score.disulfide._disulfide_energy_term.DisulfideEnergyTerm(
            param_db, device
        )

    @classmethod
    def score_types(cls):
        return cls._score_types
