from tmol.score.terms.term_creator import TermCreator, score_term_creator
from tmol.score.score_types import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class LKBallTermCreator(TermCreator):
    _score_types = [
        ScoreType.lk_ball_iso,
        ScoreType.lk_ball.ScoreType.lk_bridge,
        ScoreType.lk_bridge_uncpl,
    ]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.lk_ball.lk_ball_energy_term

        return tmol.score.lk_ball.lk_ball_energy_term.LKBallEnergyTerm(param_db, device)

    @classmethod
    def score_types(cls):
        return cls._score_types
