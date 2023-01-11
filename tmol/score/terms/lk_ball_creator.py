from tmol.score.terms.term_creator import TermCreator, score_term_creator
from tmol.score.score_types import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class LKBallTermCreator(TermCreator):
    _score_types = [ScoreType.lk_ball]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.elec.elec_energy_term

        return tmol.score.elec.elec_energy_term.ElecEnergyTerm(param_db, device)

    @classmethod
    def score_types(cls):
        return cls._score_types
