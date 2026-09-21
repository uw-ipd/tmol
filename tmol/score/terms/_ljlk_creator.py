from tmol.score.terms import TermCreator, score_term_creator
from tmol.score import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class LJLKTermCreator(TermCreator):
    """Create the Lennard-Jones and Lazaridis-Karplus energy term."""

    _score_types = [ScoreType.fa_ljatr, ScoreType.fa_ljrep, ScoreType.fa_lk]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.ljlk._ljlk_energy_term

        return tmol.score.ljlk._ljlk_energy_term.LJLKEnergyTerm(param_db, device)

    @classmethod
    def score_types(cls):
        return cls._score_types
