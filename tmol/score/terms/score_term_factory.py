from typing import Dict
import tmol.score.terms  # force factory registration of score types
from tmol.score.score_types import ScoreType


class ScoreTermFactory:
    creator_map: Dict = {}

    @classmethod
    def factory_register(cls, creator: "TermCreator"):
        sts = creator.score_types()
        for st in sts:
            assert st not in cls.creator_map
            cls.creator_map[st] = creator

    @classmethod
    def create_term_for_score_type(cls, st: ScoreType):
        creator = cls.creator_map[st]
        return creator.create_term()
