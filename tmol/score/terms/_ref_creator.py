from tmol.score.terms import TermCreator, score_term_creator
from tmol.score import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class RefTermCreator(TermCreator):
    _score_types = [ScoreType.ref]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.ref._ref_energy_term

        return tmol.score.ref._ref_energy_term.RefEnergyTerm(param_db, device)

    @classmethod
    def score_types(cls):
        return cls._score_types
