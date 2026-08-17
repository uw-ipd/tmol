from tmol.score.terms import TermCreator, score_term_creator
from tmol.score import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class ElecTermCreator(TermCreator):
    _score_types = [ScoreType.fa_elec]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.elec.elec_energy_term

        return tmol.score.elec.elec_energy_term.ElecEnergyTerm(param_db, device)

    @classmethod
    def score_types(cls):
        return cls._score_types
