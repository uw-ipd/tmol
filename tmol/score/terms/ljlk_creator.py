from tmol.score.terms.term_creator import TermCreator, score_term_creator
from tmol.score.score_types import ScoreType


print("imported ljlktermcreator")


@score_term_creator
class LJLKTermCreator(TermCreator):
    _score_types = [ScoreType.fa_lj, ScoreType.fa_lk]

    @classmethod
    def create_term(cls):
        from tmol.score.ljlk.ljlk_energy_term import LJLKEnergyTerm

        return LJLKEnergyTerm()

    @classmethod
    def score_types(cls):
        return cls._score_types
