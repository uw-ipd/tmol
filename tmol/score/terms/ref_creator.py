from tmol.score.terms.term_creator import TermCreator, score_term_creator
from tmol.score.score_types import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class RefTermCreator(TermCreator):
    _score_types = [ScoreType.ref]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.ref.ref_energy_term

        return tmol.score.ref.ref_energy_term.RefEnergyTerm(param_db, device)

    @classmethod
    def score_types(cls):
        return cls._score_types
