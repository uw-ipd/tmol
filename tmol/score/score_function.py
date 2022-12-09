import torch

from typing import Sequence
from tmol.types.torch import Tensor

from tmol.database import ParameterDatabase
from tmol.score.score_types import ScoreType

# force registration of the terms with the ScoreTermFactory
from tmol.score.terms import *  # noqa: F401, F403
from tmol.score.terms.score_term_factory import ScoreTermFactory

from tmol.pose.pose_stack import PoseStack


class ScoreFunction:
    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        self._weights = torch.zeros((ScoreType.n_score_types.value,), device=device)

        self._all_terms = []
        self._all_terms_unordered = []
        self._all_terms_out_of_date = False

        self._one_body_terms = []
        self._one_body_terms_unordered = []
        self._one_body_terms_out_of_date = False

        self._two_body_terms = []
        self._two_body_terms_unordered = []
        self._two_body_terms_out_of_date = False

        self._multi_body_terms = []
        self._multi_body_terms_unordered = []
        self._multi_body_terms_out_of_date = False

        self._weights_tensor_out_of_date = True
        self._weights_tensor = None
        self._term_for_st = [None] * ScoreType.n_score_types.value
        self._param_db = param_db
        self._device = device

    def set_weight(self, st: ScoreType, weight: float):
        if not self.score_type_covered_by_contained_term(st):
            self.retrieve_term_for_score_type(st)
        if weight == 0 and self.term_for_st_has_no_other_non_zero_weights(st):
            self.remove_term_for_score_type(st)  # TO DO!
        self._weights[st.value] = weight
        self._weights_tensor_out_of_date = True

    def score_type_covered_by_contained_term(self, st: ScoreType):
        for term in self._all_terms:
            if st in term.score_types():
                return True
        return False

    def retrieve_term_for_score_type(self, st: ScoreType):
        term = ScoreTermFactory.create_term_for_score_type(
            st, self._param_db, self._device
        )
        for st in term.score_types():
            self._term_for_st[st.value] = term
        self._all_terms_unordered.append(term)
        self._all_terms_out_of_date = True
        if term.n_bodies() == 1:
            self._one_body_terms_unordered.append(term)
            self._one_body_terms_out_of_date = True
        elif term.n_bodies() == 2:
            self._two_body_terms_unordered.append(term)
            self._two_body_terms_out_of_date = True
        else:
            self._multi_body_terms_unordered.append(term)
            self._multi_body_terms_out_of_date = True

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
        if self._all_terms_out_of_date:
            self._all_terms = self.get_sorted_terms(self._all_terms_unordered)
            self._all_terms_out_of_date = False

        return self._all_terms

    def one_body_terms(self):
        if self._one_body_terms_out_of_date:
            self._one_body_terms = self.get_sorted_terms(self._one_body_terms_unordered)
            self._one_body_terms_out_of_date = False

        return self._one_body_terms

    def two_body_terms(self):
        if self._two_body_terms_out_of_date:
            self._two_body_terms = self.get_sorted_terms(self._two_body_terms_unordered)
            self._two_body_terms_out_of_date = False

        return self._two_body_terms

    def multi_body_terms(self):
        if self._multi_body_terms_out_of_date:
            self._multi_body_terms = self.get_sorted_terms(
                self._multi_body_terms_unordered
            )
            self._multi_body_terms_out_of_date = False

        return self._multi_body_terms

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        """Create an object designed to evaluate the score of a set of Poses
        repeatedly as the Poses change their conformation, e.g., as in
        minimization. This object will derive from torch.nn.Module and
        it will contain a set of objects rendered by the ScoreFunction's
        terms that themselves are derived from torch.nn.Module
        """
        self.pre_work_initialization(pose_stack)
        term_modules = [
            t.render_whole_pose_scoring_module(pose_stack) for t in self.all_terms()
        ]
        return WholePoseScoringModule(self.weights_tensor(), term_modules)

    def pre_work_initialization(self, pose_stack: PoseStack):
        for block_type in pose_stack.packed_block_types.active_block_types:
            for energy_term in self.all_terms():
                energy_term.setup_block_type(block_type)
        for energy_term in self.all_terms():
            energy_term.setup_packed_block_types(pose_stack.packed_block_types)
        for energy_term in self.all_terms():
            energy_term.setup_poses(pose_stack)

    def weights_tensor(self):
        if self._weights_tensor_out_of_date:
            self._weights_tensor = torch.tensor(
                [
                    self._weights[st.value]
                    for term in self.all_terms()
                    for st in term.score_types()
                ],
                dtype=torch.float32,
                device=self._device,
            )
            self._weights_tensor_out_of_date = False
        return self._weights_tensor

    @staticmethod
    def get_sorted_terms(term_list):
        sorted_term_list = []
        term_covered = [False] * ScoreType.n_score_types.value
        terms_by_st = [None] * ScoreType.n_score_types.value
        for term in term_list:
            for term_st in term.score_types():
                terms_by_st[term_st.value] = term

        for st_ind in range(ScoreType.n_score_types.value):
            if terms_by_st[st_ind] is not None:
                already_covered = False
                term = terms_by_st[st_ind]
                for term_st in term.score_types():
                    if term_covered[term_st.value]:
                        already_covered = True
                        break
                if not already_covered:
                    sorted_term_list.append(term)
                    for term_st in term.score_types():
                        term_covered[term_st.value] = True
        return sorted_term_list


class WholePoseScoringModule:
    def __init__(
        self, weights: Tensor[torch.float32][:], term_modules: Sequence[torch.nn.Module]
    ):
        # super(WholePoseScoringModule, self).__init__()
        self.weights = torch.nn.Parameter(weights.unsqueeze(1), requires_grad=False)
        self.term_modules = term_modules

    def __call__(self, coords):
        all_scores = torch.cat([term(coords) for term in self.term_modules], dim=0)
        return torch.sum(self.weights * all_scores)
