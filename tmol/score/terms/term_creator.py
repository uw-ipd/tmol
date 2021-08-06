import torch

from tmol.score.terms.score_term_factory import ScoreTermFactory
from tmol.database import ParameterDatabase


class TermCreator:
    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        raise NotImplementedError()

    @classmethod
    def score_types(cls):
        raise NotImplementedError()


def score_term_creator(cls):
    ScoreTermFactory.factory_register(cls)
