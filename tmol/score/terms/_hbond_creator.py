from tmol.score.terms import TermCreator, score_term_creator
from tmol.score import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class HBondTermCreator(TermCreator):
    """Create the hydrogen-bond energy term."""

    _score_types = [ScoreType.hbond]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.hbond._hbond_energy_term

        return tmol.score.hbond._hbond_energy_term.HBondEnergyTerm(param_db, device)

    @classmethod
    def score_types(cls):
        return cls._score_types
