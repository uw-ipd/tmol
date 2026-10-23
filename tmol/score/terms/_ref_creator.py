from tmol.score.terms._term_creator import TermCreator, _score_term_creator
from tmol.score._score_types import ScoreType
from tmol.database import ParameterDatabase
import torch


@_score_term_creator
class RefTermCreator(TermCreator):
    """Create the residue reference-energy term."""

    _score_types = [ScoreType.ref]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.ref._ref_energy_term

        return tmol.score.ref._ref_energy_term.RefEnergyTerm(param_db, device)

    @classmethod
    def score_types(cls):
        return cls._score_types
