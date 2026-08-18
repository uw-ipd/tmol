import torch

from .._energy_term import EnergyTerm

from tmol.database import ParameterDatabase

from tmol.chemical import RefinedResidueType
from tmol.pose import PackedBlockTypes
from tmol.pose import PoseStack


class RefEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(RefEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.ref_weights = param_db.scoring.ref.weights
        self.weights_override = None
        self.soft_rep = False
        self.device = device

    @classmethod
    def class_name(cls):
        return "Ref"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._ref_creator

        return tmol.score.terms._ref_creator.RefTermCreator.score_types()

    def n_bodies(self):
        return 1

    def set_options(self, options: dict):
        if "ref_weights" in options:
            self.weights_override = options["ref_weights"]

    def _resolved_weights(self) -> dict:
        """The ref-weight map this term scores with (override beats the db default)."""
        return self.weights_override if self.weights_override else self.ref_weights

    def setup_block_type(self, block_type: RefinedResidueType):
        super(RefEnergyTerm, self).setup_block_type(block_type)

        # ``ref_weight`` is score-function-specific: ``set_options`` can override
        # the database defaults per score function. Block types are shared/cached
        # across score functions, so the cache must be keyed by the resolved
        # weight map -- otherwise a second score function (e.g. beta_soft after
        # beta2016) silently reuses the first's ref weights. (See the matching
        # guard in ``setup_packed_block_types``.)
        src = self._resolved_weights()
        if getattr(block_type, "_ref_weight_src", None) is src:
            return

        ref_weight = 0.0
        if block_type.base_name in self.ref_weights:
            ref_weight = src[block_type.base_name]

        setattr(block_type, "ref_weight", ref_weight)
        setattr(block_type, "_ref_weight_src", src)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(RefEnergyTerm, self).setup_packed_block_types(packed_block_types)

        # Keyed by the resolved weight map for the same reason as
        # ``setup_block_type``: the PackedBlockTypes is cached and shared across
        # score functions, so reusing a stale ``ref_weights`` tensor would leak
        # one score function's reference energies into another.
        src = self._resolved_weights()
        if (
            hasattr(packed_block_types, "ref_weights")
            and getattr(packed_block_types, "_ref_weights_src", None) is src
        ):
            return

        ref_weights = []
        for bt in packed_block_types.active_block_types:
            ref_weights += [bt.ref_weight]

        ref_weights = torch.as_tensor(
            ref_weights, dtype=torch.float32, device=self.device
        )

        setattr(packed_block_types, "ref_weights", ref_weights)
        setattr(packed_block_types, "_ref_weights_src", src)

    def setup_poses(self, poses: PoseStack):
        super(RefEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        return eval_ref_energy_for_pose

    def get_rotamer_score_term_function(self):
        return eval_ref_energy_for_rotamers

    def get_score_term_attributes(self, pose_stack):
        # ref depends only on block type, so per-block energies are constant for
        # this pose stack; the rotamer path indexes by block type and needs the
        # raw weights instead.
        ref_weights = pose_stack.packed_block_types.ref_weights
        bt = pose_stack.block_type_ind64
        weights = ref_weights[bt.clamp_min(0)]
        block_ref = torch.where(bt >= 0, weights, torch.zeros_like(weights))
        return [block_ref, ref_weights]


def eval_ref_energy_for_pose(
    # common args
    _rot_coords,
    _rot_coord_offset,
    _pose_ind_for_atom,
    _first_rot_for_block,
    _first_rot_block_type,
    _block_ind_for_rot,
    _pose_ind_for_rot,
    _block_type_ind_for_rot,
    _n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _max_n_rots_per_pose,
    block_ref,
    _ref_weights,
    output_block_pair_energies: bool,
):
    score = block_ref

    if output_block_pair_energies:
        score = torch.diag_embed(score)
    else:
        # for each pose, sum up the block scores
        score = torch.sum(score, 1)

    # wrap this all in an extra dim (the output expects an outer dim to separate sub-terms)
    score = torch.unsqueeze(score, 0)

    score.requires_grad = True  # a bit of a hack to make the benchmark test not error out because there are no grads

    return score, None


def eval_ref_energy_for_rotamers(
    # common args
    rot_coords,
    _rot_coord_offset,
    _pose_ind_for_atom,
    _first_rot_for_block,
    _first_rot_block_type,
    _block_ind_for_rot,
    pose_ind_for_rot,
    block_type_ind_for_rot,
    n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _max_n_rots_per_pose,
    _block_ref,
    ref_weights,
    output_block_pair_energies: bool,
):
    block_type_ind_for_rot64 = block_type_ind_for_rot.to(torch.int64)

    # fill out the scores for the real blocks by dereferencing the block types into the ref weights
    dtype = ref_weights.dtype
    assert rot_coords.dtype == dtype
    is_real_rot = block_type_ind_for_rot64 >= 0
    rot_ref = ref_weights[block_type_ind_for_rot64.clamp_min(0)]
    rotamer_scores = torch.where(is_real_rot, rot_ref, torch.zeros_like(rot_ref))
    device = rot_coords.device

    if output_block_pair_energies:
        n_rotamers = pose_ind_for_rot.shape[0]
        indices = torch.zeros((3, n_rotamers), dtype=torch.int32, device=device)
        indices[0, :] = pose_ind_for_rot
        rot_ind = torch.arange(n_rotamers, dtype=torch.int32, device=device)
        indices[1, :] = rot_ind
        indices[2, :] = rot_ind
        output_scores = rotamer_scores
    else:
        # for each pose, sum up the block scores
        pose_ind_for_rot64 = pose_ind_for_rot.to(torch.int64)
        output_scores = torch.zeros_like((n_rots_for_pose), dtype=dtype)
        output_scores.index_add_(0, pose_ind_for_rot64, rotamer_scores)
        indices = torch.zeros((0,), dtype=torch.int32, device=device)
    output_scores = output_scores.unsqueeze(0)
    output_scores.requires_grad = True  # a bit of a hack to make the benchmark test not error out because there are no grads
    return output_scores, indices
