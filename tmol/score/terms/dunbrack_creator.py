from tmol.score.terms import TermCreator, score_term_creator
from tmol.score import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class DunbrackTermCreator(TermCreator):
    _score_types = [
        ScoreType.dunbrack_rot,
        ScoreType.dunbrack_rotdev,
        ScoreType.dunbrack_semirot,
    ]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.dunbrack.dunbrack_energy_term

        return tmol.score.dunbrack.dunbrack_energy_term.DunbrackEnergyTerm(
            param_db, device
        )

    @classmethod
    def score_types(cls):
        return cls._score_types
