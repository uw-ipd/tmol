from tmol.score.terms import TermCreator, score_term_creator
from tmol.score import ScoreType
from tmol.database import ParameterDatabase
import torch


@score_term_creator
class CartBondedTermCreator(TermCreator):
    """Create the Cartesian bonded-geometry energy term."""

    _score_types = [
        ScoreType.cart_lengths,
        ScoreType.cart_angles,
        ScoreType.cart_torsions,
        ScoreType.cart_impropers,
        ScoreType.cart_hxltorsions,
    ]

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        import tmol.score.cartbonded._cartbonded_energy_term

        return tmol.score.cartbonded._cartbonded_energy_term.CartBondedEnergyTerm(
            param_db, device
        )

    @classmethod
    def score_types(cls):
        return cls._score_types
