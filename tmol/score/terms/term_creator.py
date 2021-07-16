from tmol.score.terms.score_term_factory import ScoreTermFactory


class TermCreator:
    @classmethod
    def create_term(cls):
        raise NotImplementedError()

    @classmethod
    def score_types(cls):
        raise NotImplementedError()


def score_term_creator(cls):
    ScoreTermFactory.factory_register(cls)
