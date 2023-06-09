import torch

from tmol.score.omega.potentials.compiled import omega_pose_scores


class OmegaWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_types,
        pose_stack_inter_block_connections,
        bt_omega_quad_uaids,
        bt_atom_downstream_of_conn,
        global_params,
    ):
        super(OmegaWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_inter_block_connections = _p(pose_stack_inter_block_connections)
        self.bt_omega_quad_uaids = _p(bt_omega_quad_uaids)
        self.bt_atom_downstream_of_conn = _p(bt_atom_downstream_of_conn)

        self.global_params = _p(torch.stack(_t([global_params.K]), dim=1))

    def forward(self, coords):
        return omega_pose_scores(
            coords,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_types,
            self.pose_stack_inter_block_connections,
            self.bt_omega_quad_uaids,
            self.bt_atom_downstream_of_conn,
            self.global_params,
        )
