import torch

from tmol.score.terms.score_term_factory import ScoreTermFactory
from tmol.database import ParameterDatabase


class TermCreator:
    """Base class for registering score terms with the ScoreTermFactory.

    To add a new term,

      - add one or more new entries to the tmol.score.score_types enumeration
      - derive a new subclass of TermCreator and put it in this directory
        (the term itself should be implemented in a different directory)
      - the new TermCreator subclass needs to define two methods,
        create_term and score_types
      - create_term should instantiate the term
      - score_types should return a list of the elements of the score_types
        enumeration that the term implements in the order that the term
        will report them
    """

    @classmethod
    def create_term(cls, param_db: ParameterDatabase, device: torch.device):
        raise NotImplementedError()

    @classmethod
    def score_types(cls):
        raise NotImplementedError()


def score_term_creator(cls):
    ScoreTermFactory.factory_register(cls)
    return cls
