from tmol.score.score_types import ScoreType
from tmol.score.terms.score_term_factory import ScoreTermFactory


class ScoreFunction:
    def __init__(self, device: torch.device):
        self._weights = torch.zeros((ScoreType.n_score_types,), device=device)
        self._all_terms = []
        self._one_body_terms = []
        self._two_body_terms = []
        self._multi_body_terms = []
        self._weights_tensor_out_of_date = True
        self._weights_tensor = None

    def set_weight(self, st: ScoreType, weight: float):
        if not self.score_type_covered_by_contained_term(st):
            self.retrieve_term_for_score_type(st)
        if weight == 0 and self.term_for_st_has_no_other_non_zero_weights(st):
            self.remove_term_for_score_type(st)
        self._weights[int(st)] = weight
        self._weights_tensor_out_of_date = True

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

        Do not modify this list directly
        """
        return self._all_terms

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        """Create an object designed to evaluate the score of a set of Poses
        repeatedly as the Poses change their conformation, e.g., as in
        minimization. This object will derive from torch.nn.Module and
        it will contain a set of objects rendered by the ScoreFunction's
        terms that themselves are derived from torch.nn.Module
        """
        self.pre_work_initialization(self, pose_stack)
        term_modules = [
            t.render_whole_pose_scoring_module(pose_stack) for t in self._all_terms
        ]
        return WholePoseScoringModule(self.weights_tensor(), term_modules)

    def pre_work_initialization(self, pose_stack: PoseStack):
        for block_type in pose_stack.packed_block_types.active_block_types:
            for energy_term in self._all_terms:
                energy_term.setup_block_type(block_type)
        for energy_term in self._all_terms:
            energy_term.setup_packed_block_types(pose_stack.packed_block_types)
            energy_term.setup_poses(pose_stack)

    def weights_tensor(self):
        if self._weights_out_of_date:
            self._weights_tensor = torch.tensor(
                [
                    self.weights[st]
                    for term in self._all_terms
                    for st in term.score_types()
                ],
                dtype=torch.float32,
                device=self.device,
            )
            self._weights_tensor_out_of_date = False
        return self._weights_tensor
