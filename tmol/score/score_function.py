import torch
import yaml
import logging

from typing import Dict, Sequence
from tmol.types.torch import Tensor

from tmol.database import ParameterDatabase
from tmol.score.score_types import ScoreType

# force registration of the terms with the ScoreTermFactory
from tmol.score.terms import *  # noqa: F401, F403
from tmol.score.terms.score_term_factory import ScoreTermFactory

from tmol.pose.pose_stack import PoseStack

logger = logging.getLogger(__name__)

# Current .sfxn (score function weights YAML) format version.  Bump the major
# version on breaking schema changes; bump the minor version on
# backward-compatible additions.  The version string is written into every
# .sfxn file and checked on load.
SFXN_FORMAT_VERSION: str = "1.0"


class ScoreFunction:
    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        self._weights = torch.zeros((ScoreType.n_score_types.value,), device=device)

        self._all_terms = []
        self._all_terms_unordered = []
        self._all_terms_out_of_date = False

        self._all_score_types = []

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

        self.term_options = {}

    def set_weight(self, st: ScoreType, weight: float):
        if not self.score_type_covered_by_contained_term(st):
            self.retrieve_term_for_score_type(st)
        if weight == 0 and self.term_for_st_has_no_other_non_zero_weights(st):
            self.remove_term_for_score_type(st)  # TO DO!
        self._weights[st.value] = weight
        self._weights_tensor_out_of_date = True

    def get_weight(self, st: ScoreType):
        return self._weights[st.value]

    def score_type_covered_by_contained_term(self, st: ScoreType):
        for term in self._all_terms:
            if st in term.score_types():
                return True
        return False

    def retrieve_term_for_score_type(self, st: ScoreType):
        term = ScoreTermFactory.create_term_for_score_type(
            st, self._param_db, self._device
        )
        # sanity check: if the ScoreTermFactory returns the wrong term,
        # we want to know
        assert st in term.score_types()
        for tst in term.score_types():
            self._term_for_st[tst.value] = term
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
        term = self._term_for_st[st.value]
        for st2 in term.score_types():
            if st2 == st:
                continue
            if self._weights[st2.value] != 0:
                return True
        return False

    def all_terms(self):
        """Grant read access to the list of terms.

        Do not modify this list directly
        """
        if self._all_terms_out_of_date:
            self._all_terms, self._all_score_types = self.get_sorted_terms(
                self._all_terms_unordered
            )
            self._all_terms_out_of_date = False

        return self._all_terms

    def all_score_types(self):
        if self._all_terms_out_of_date:
            self._all_terms, self._all_score_types = self.get_sorted_terms(
                self._all_terms_unordered
            )
            self._all_terms_out_of_date = False

        return self._all_score_types

    def one_body_terms(self):
        if self._one_body_terms_out_of_date:
            self._one_body_terms, _ = self.get_sorted_terms(
                self._one_body_terms_unordered
            )
            self._one_body_terms_out_of_date = False

        return self._one_body_terms

    def two_body_terms(self):
        if self._two_body_terms_out_of_date:
            self._two_body_terms, _ = self.get_sorted_terms(
                self._two_body_terms_unordered
            )
            self._two_body_terms_out_of_date = False

        return self._two_body_terms

    def multi_body_terms(self):
        if self._multi_body_terms_out_of_date:
            self._multi_body_terms, _ = self.get_sorted_terms(
                self._multi_body_terms_unordered
            )
            self._multi_body_terms_out_of_date = False

        return self._multi_body_terms

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        """Create an object designed to evaluate the score of a set of Poses
        repeatedly as the Poses change their conformation, e.g., as in
        minimization. This object will derive from torch.nn.Module and
        it will contain a set of objects rendered by the ScoreFunction's
        terms that themselves are derived from torch.nn.Module. This
        object's __call__ will return a tensor of weighted energies of
        shape (n_poses,).
        """
        self.pre_work_initialization(pose_stack)
        term_modules = [
            t.render_whole_pose_scoring_module(pose_stack) for t in self.all_terms()
        ]
        return WholePoseScoringModule(self.weights_tensor(), term_modules)

    def render_block_pair_scoring_module(self, pose_stack: PoseStack):
        """Create an object designed to evaluate the score of a set of Poses
        repeatedly as the Poses change their conformation, e.g., as in
        minimization. This object will derive from torch.nn.Module and
        it will contain a set of objects rendered by the ScoreFunction's
        terms that themselves are derived from torch.nn.Module. This
        object's __call__ will return a tensor of weighted energies of
        shape (n_poses, max_n_blocks, max_n_blocks).
        """
        self.pre_work_initialization(pose_stack)
        term_modules = [
            t.render_block_pair_scoring_module(pose_stack) for t in self.all_terms()
        ]
        return BlockPairScoringModule(self.weights_tensor(), term_modules)

    def render_rotamer_scoring_module(
        self, pose_stack: PoseStack, rotamer_set: "RotamerSet"  # noqa: F405
    ):
        """Create an object designed to evaluate the score a RotamerSet
        repeatedly as the Poses change their conformation, e.g., as in
        minimization. This object will derive from torch.nn.Module and
        it will contain a set of objects rendered by the ScoreFunction's
        terms that themselves are derived from torch.nn.Module. This
        object's __call__ will return a tensor of weighted energies of
        shape (n_poses, max_n_blocks, max_n_blocks).
        """
        self.pre_work_initialization(pose_stack)
        term_modules = [
            t.render_rotamer_scoring_module(pose_stack, rotamer_set)
            for t in self.all_terms()
        ]
        return RotamerScoringModule(self.weights_tensor(), term_modules)

    def pre_work_initialization(self, pose_stack: PoseStack):
        # set_options must be first, since some of the logic that follows it
        # may depend on the options
        for energy_term in self.all_terms():
            energy_term.set_options(self.term_options)

        for block_type in pose_stack.packed_block_types.active_block_types:
            for energy_term in self.all_terms():
                energy_term.setup_block_type(block_type)
        for energy_term in self.all_terms():
            energy_term.setup_packed_block_types(pose_stack.packed_block_types)
        for energy_term in self.all_terms():
            energy_term.setup_poses(pose_stack)

    def set_option(self, key: str, value):
        """Set an option for all energy terms.

        Options are passed to each energy term's set_options method
        as a dictionary during pre_work_initialization.
        """
        self.term_options[key] = value

    def set_options(self, options: Dict):
        """Set the score function options by a dict.

        This replaces the options dict entirely - any previous values
        are gone.
        """
        self.term_options = options

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

    @classmethod
    def from_sfxn_file(cls, path, param_db, device):
        """Create a ScoreFunction from a YAML weights file.

        Args:
            path: Path to a YAML file containing a ``weights`` dict mapping
                score type names (as in ``ScoreType``) to their weights, as well
                as any other options to configure the score function.
            param_db: ParameterDatabase instance.
            device: Target torch device.

        Returns:
            Configured ScoreFunction with all weights from the file applied.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        # --- .sfxn format version check ---
        file_version = data.get("version")
        if file_version is None:
            raise ValueError(
                f"{path}: no 'version' field found in .sfxn file. "
                f"Current format version is {SFXN_FORMAT_VERSION}. "
                f"Regenerate the file with the current version."
            )
        else:
            file_version = str(file_version)
            file_major = file_version.split(".")[0]
            current_major = SFXN_FORMAT_VERSION.split(".")[0]
            if file_major != current_major:
                raise ValueError(
                    f"{path}: .sfxn format version {file_version} is incompatible "
                    f"with the current format version {SFXN_FORMAT_VERSION}. "
                    f"Regenerate the file with the current writer."
                )
            if file_version != SFXN_FORMAT_VERSION:
                logger.info(
                    "%s: .sfxn format version %s differs from current %s "
                    "(backward-compatible minor version change)",
                    path,
                    file_version,
                    SFXN_FORMAT_VERSION,
                )

        sfxn = cls(param_db, device)
        for name, weight in data["weights"].items():
            sfxn.set_weight(getattr(ScoreType, name), weight)
        if "options" in data:
            sfxn.set_options(data["options"])
        return sfxn

    @staticmethod
    def get_sorted_terms(term_list):
        sorted_term_list = []
        sorted_score_type_list = []
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
                        sorted_score_type_list.append(term_st)
        return sorted_term_list, sorted_score_type_list


class WholePoseScoringModule:
    def __init__(
        self,
        weights: Tensor[torch.float32][:],
        term_modules: Sequence[torch.nn.Module],
    ):
        self.weights = torch.nn.Parameter(weights.unsqueeze(1), requires_grad=False)
        self.term_modules = term_modules

    def __call__(self, coords, sum_terms=True, apply_weights=True):
        unweighted = self.unweighted_scores(coords)
        weighted = self.weights * unweighted if apply_weights else unweighted
        summed = torch.sum(weighted, dim=0) if sum_terms else weighted

        return summed

    def unweighted_scores(self, coords):
        return torch.cat([term(coords) for term in self.term_modules], dim=0)


class BlockPairScoringModule:
    def __init__(
        self,
        weights: Tensor[torch.float32][:],
        term_modules: Sequence[torch.nn.Module],
    ):
        self.weights = torch.nn.Parameter(
            weights.unsqueeze(1).unsqueeze(1).unsqueeze(1), requires_grad=False
        )
        self.term_modules = term_modules

    def __call__(self, coords, sum_terms=True, apply_weights=True):
        unweighted = self.unweighted_scores(coords)
        weighted = self.weights * unweighted if apply_weights else unweighted
        summed = torch.sum(weighted, dim=0) if sum_terms else weighted

        return summed

    def unweighted_scores(self, coords):
        return torch.cat([term(coords) for term in self.term_modules], dim=0)


class RotamerScoringModule:
    def __init__(
        self,
        weights: Tensor[torch.float32][:],
        term_modules: Sequence[torch.nn.Module],
    ):
        self.weights = torch.nn.Parameter(
            weights.view(-1, 1, 1, 1), requires_grad=False
        )
        self.term_modules = term_modules

    def __call__(self, coords):
        # Accumulate weighted values and their indices across all terms at the
        # dense [nnz] level.  This avoids torch.stack on sparse tensors, which
        # previously created a [n_subterms, n_poses, n_rots, n_rots] 4D sparse
        # tensor whose index storage grew as n_subterms × nnz × 4 int32.
        all_values = []
        all_indices = []
        n_poses = None
        n_rots = None
        weights_offset = 0

        for term in self.term_modules:
            scores, indices = term.forward(coords)  # [n_subterms, nnz], [3, nnz]
            n_subterms = scores.shape[0]

            # Apply per-subterm weights and sum to [nnz] — no sparse tensor yet.
            w = self.weights[weights_offset : weights_offset + n_subterms, 0, 0, 0]
            weighted_values = (w[:, None] * scores).sum(dim=0)

            all_values.append(weighted_values)
            all_indices.append(indices)
            weights_offset += n_subterms

            if n_poses is None:
                n_poses = term.n_poses
                n_rots = term.n_rots

        if n_poses is None:
            # No terms at all
            return torch.sparse_coo_tensor(
                torch.zeros((3, 0), dtype=torch.int32, device=coords.device),
                torch.zeros(0, dtype=torch.float32, device=coords.device),
                size=(0, 0, 0),
            )

        combined_values = torch.cat(all_values)
        combined_indices = torch.cat(all_indices, dim=1)
        return torch.sparse_coo_tensor(
            combined_indices,
            combined_values,
            size=(n_poses, n_rots, n_rots),
        )
