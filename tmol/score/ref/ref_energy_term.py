import torch

from ..energy_term import EnergyTerm

from tmol.database import ParameterDatabase

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack


class RefEnergyTerm(EnergyTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(RefEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.ref_weights = param_db.scoring.ref.weights
        self.device = device

    @classmethod
    def class_name(cls):
        return "Ref"

    @classmethod
    def score_types(cls):
        import tmol.score.terms.ref_creator

        return tmol.score.terms.ref_creator.RefTermCreator.score_types()

    def n_bodies(self):
        return 1

    def setup_block_type(self, block_type: RefinedResidueType):
        super(RefEnergyTerm, self).setup_block_type(block_type)

        if hasattr(block_type, "ref_weight"):
            return

        ref_weight = 0.0

        if block_type.base_name in self.ref_weights:
            ref_weight = self.ref_weights[block_type.base_name]

        setattr(block_type, "ref_weight", ref_weight)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(RefEnergyTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "ref_weights"):
            return

        ref_weights = []
        for bt in packed_block_types.active_block_types:
            ref_weights += [bt.ref_weight]

        ref_weights = torch.as_tensor(
            ref_weights, dtype=torch.float32, device=self.device
        )

        setattr(packed_block_types, "ref_weights", ref_weights)

    def setup_poses(self, poses: PoseStack):
        super(RefEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        return eval_ref_energy_for_pose

    def get_rotamer_score_term_function(self):
        return eval_ref_energy_for_rotamers

    def get_score_term_attributes(self, pose_stack):
        # print("arrived at get_score_term_attributes")
        atts = [pose_stack.packed_block_types.ref_weights]
        # print("n atts:", len(atts))
        return atts


def eval_ref_energy_for_pose(
    # common args
    rot_coords,
    _rot_coord_offset,
    _pose_ind_for_atom,
    first_rot_for_block,
    _first_rot_block_type,
    _block_ind_for_rot,
    pose_ind_for_rot,
    block_type_ind_for_rot,
    _n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _max_n_rots_per_pose,
    ref_weights,
    output_block_pair_energies: bool,
):
    n_poses = first_rot_for_block.shape[0]
    max_n_blocks = first_rot_for_block.shape[1]
    block_type_ind_for_rot = block_type_ind_for_rot.view(n_poses, max_n_blocks)
    # block_type_ind_for_rot64 = block_type_ind_for_rot.to(torch.int64)

    # fill our per-block ref scores with zeros to start
    score = torch.zeros_like(block_type_ind_for_rot, dtype=rot_coords.dtype)

    # grab the indices of any non-negative (real) blocks
    real_blocks = block_type_ind_for_rot >= 0

    # fill out the scores for the real blocks by dereferencing the block types into the ref weights
    score[real_blocks] = torch.index_select(
        ref_weights, 0, block_type_ind_for_rot[real_blocks]
    )

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
    ref_weights,
    output_block_pair_energies: bool,
):
    block_type_ind_for_rot64 = block_type_ind_for_rot.to(torch.int64)

    # fill out the scores for the real blocks by dereferencing the block types into the ref weights
    dtype = ref_weights.dtype
    assert rot_coords.dtype == dtype
    is_real_rot = block_type_ind_for_rot64 >= 0
    rotamer_scores = torch.index_select(
        ref_weights, 0, block_type_ind_for_rot64[is_real_rot]
    )
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
        output_scores.index_add_(0, pose_ind_for_rot64[is_real_rot], rotamer_scores)
        indices = torch.zeros((0,), dtype=torch.int32, device=device)
    output_scores = output_scores.unsqueeze(0)
    output_scores.requires_grad = True  # a bit of a hack to make the benchmark test not error out because there are no grads
    return output_scores, indices
