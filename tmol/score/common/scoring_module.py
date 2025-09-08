import torch

# from tmol.score.hbond.potentials.compiled import hbond_pose_scores
from tmol.score.common.convert_float64 import convert_float64


class ScoringModule(torch.nn.Module):
    def __init__(
        self,
        term_parameters,
        term_score_poses,
    ):
        super(ScoringModule, self).__init__()

        self.term_parameters = []

        self.add_parameters(self.term_parameters, term_parameters)

        self.term_score_poses = term_score_poses

    def format_arguments(self, coords, block_pair_scoring):
        params = (
            [coords.flatten(start_dim=0, end_dim=-2)]
            + self.common_parameters
            + self.term_parameters
            + [block_pair_scoring]
        )

        if coords.dtype == torch.float64:
            convert_float64(params)

        return params

    def add_parameters(self, table, params):
        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        for param in params:
            table += [_p(param) if type(param) is torch.Tensor else param]


class PoseScoringModule(ScoringModule):
    def __init__(
        self,
        pose_stack,
        term_parameters,
        term_score_poses,
    ):
        super(PoseScoringModule, self).__init__(term_parameters, term_score_poses)

        self.common_parameters = []

        self.add_parameters(
            self.common_parameters,
            [
                pose_stack.rot_coord_offset,
                pose_stack.pose_ind_for_atom,
                pose_stack.first_rot_for_block,
                pose_stack.first_rot_for_block,
                pose_stack.block_ind_for_rot,
                pose_stack.pose_ind_for_rot,
                pose_stack.block_type_ind_for_rot,
                pose_stack.n_rots_for_pose,
                pose_stack.rot_offset_for_pose,
                pose_stack.n_rots_for_block,
                pose_stack.rot_offset_for_block,
                pose_stack.max_n_rots_per_pose,
            ],
        )


class WholePoseScoringModule(PoseScoringModule):
    def forward(
        self,
        coords,
    ):
        scores, _ = self.term_score_poses(*self.format_arguments(coords, False))

        return scores


class BlockPairScoringModule(PoseScoringModule):
    def forward(
        self,
        coords,
    ):
        scores, indices = self.term_score_poses(*self.format_arguments(coords, True))

        sparse_result = torch.stack(
            [
                torch.sparse_coo_tensor(indices, scores[subterm, :])
                for subterm in range(scores.size(0))
            ]
        )

        return sparse_result


class RotamerScoringModule(ScoringModule):
    def __init__(
        self,
        rotamer_set,
        term_parameters,
        term_score_poses,
    ):
        super(ScoringModule, self).__init__(term_parameters, term_score_poses)

        self.common_parameters = []

        self.add_parameters(
            self.common_parameters,
            [
                i.to(torch.int32)
                for i in [
                    rotamer_set.rot_coord_offset,
                    rotamer_set.first_rot_for_block,
                    rotamer_set.first_rot_for_block,
                    rotamer_set.block_ind_for_rot,
                    rotamer_set.pose_ind_for_rot,
                    rotamer_set.block_type_ind_for_rot,
                    rotamer_set.n_rots_for_pose,
                    rotamer_set.rot_offset_for_pose,
                    rotamer_set.n_rots_for_block,
                    rotamer_set.rot_offset_for_block,
                    rotamer_set.max_n_rots_per_pose,
                ]
            ],
        )

    def forward(
        self,
        coords,
    ):
        scores, indices = self.term_score_poses(*self.format_arguments(coords, True))

        return scores
