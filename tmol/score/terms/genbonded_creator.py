from tmol.score.terms.term_creator import TermCreator, score_term_creator
from tmol.score.score_types import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class GenBondedTermCreator(TermCreator):
    _score_types = [
        ScoreType.gen_torsions,
    ]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.genbonded.genbonded_energy_term

        return tmol.score.genbonded.genbonded_energy_term.GenBondedEnergyTerm(
            param_db, device
        )

    @classmethod
    def score_types(cls):
        return cls._score_types
