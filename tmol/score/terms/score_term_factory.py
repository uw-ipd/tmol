import torch
import tmol.score.terms  # force factory registration of score types

from typing import Dict
from tmol.database import ParameterDatabase
from tmol.score.score_types import ScoreType


class ScoreTermFactory:
    """Factory for the creation of EnergyTerms

    This class uses import-time factory registration to discover the set
    of TermCreators that live in the same directory as it. To register
    a new TermCreator, simply put the term creator in this directory.
    """

    creator_map: Dict = {}

    @classmethod
    def factory_register(cls, creator: "TermCreator"):
        sts = creator.score_types()
        for st in sts:
            assert st not in cls.creator_map
            cls.creator_map[st] = creator

    @classmethod
    def create_term_for_score_type(
        cls, st: ScoreType, param_db: ParameterDatabase, device: torch.device
    ):
        creator = cls.creator_map[st]
        return creator.create_term(param_db, device)
