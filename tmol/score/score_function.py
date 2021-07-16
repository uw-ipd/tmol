from tmol.score.score_types import ScoreType
from tmol.score.terms.score_term_factory import ScoreTermFactory


class ScoreFunction:
    def __init__(self, device: torch.device):
        self._weights = torch.zeros((ScoreType.n_score_types,), device=device)
        self._all_terms = []
        self._one_body_terms = []
        self._two_body_terms = []
        self._multi_body_terms = []

    def set_weight(self, st: ScoreType, weight: float):
        if not self.score_type_covered_by_contained_term(st):
            self.retrieve_term_for_score_type(st)
        if weight == 0 and self.term_for_st_has_no_other_non_zero_weights(st):
            self.remove_term_for_score_type(st)
        self._weights[int(st)] = weight

    def score_term_covered_by_contained_term(self, st: ScoreType):
        for term in self._all_terms:
            if st in term.score_types:
                return True
        return False

    def retrieve_term_for_score_type(self, st: ScoreType):
        term = TermFactory.create_term_for_score_type(st)
        self._all_terms.append(term)
        if term.n_bodies() == 1:
            self._one_body_terms.append(term)
        elif term.n_bodies() == 2:
            self._two_body_terms.append(term)
        else:
            self._multi_body_terms.append(term)

    def term_for_st_has_no_other_non_zero_weights(self, st: ScoreType):
        term = self.term_for_st(st)
        for st2 in term.score_types:
            if st2 == st:
                continue
            if self._weights[st2] != 0:
                return True
        return False

    def all_terms(self):
        """Grant read access to the list of terms.
        Do not modify this list directly"""
        return self._all_terms
