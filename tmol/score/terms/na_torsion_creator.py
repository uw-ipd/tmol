from tmol.score.terms.term_creator import TermCreator, score_term_creator
from tmol.score.score_types import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class NaTorsionTermCreator(TermCreator):
    _score_types = [ScoreType.na_torsion, ScoreType.na_torsion_well]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.na_torsion.na_torsion_energy_term

        return tmol.score.na_torsion.na_torsion_energy_term.NaTorsionEnergyTerm(
            param_db, device
        )

    @classmethod
    def score_types(cls):
        return cls._score_types
