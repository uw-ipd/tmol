from tmol.score.terms._term_creator import TermCreator, score_term_creator
from tmol.score._score_types import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class GenBondedTermCreator(TermCreator):
    """Create the general bonded-torsion energy term."""

    _score_types = [
        ScoreType.gen_torsions,
    ]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.genbonded._genbonded_energy_term

        return tmol.score.genbonded._genbonded_energy_term.GenBondedEnergyTerm(
            param_db, device
        )

    @classmethod
    def score_types(cls):
        return cls._score_types
