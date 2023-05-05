import torch

from tmol.score.disulfide.potentials.compiled import disulfide_pose_scores


class DisulfideWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_types,
        pose_stack_inter_block_connections,
        bt_disulfide_conns,
        bt_atom_downstream_of_conn,
        global_params,
    ):
        super(DisulfideWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_inter_block_connections = _p(pose_stack_inter_block_connections)
        self.bt_disulfide_conns = _p(bt_disulfide_conns)
        self.bt_atom_downstream_of_conn = _p(bt_atom_downstream_of_conn)

        self.global_params = _p(global_params)

    def forward(self, coords):
        return disulfide_pose_scores(
            coords,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_types,
            self.pose_stack_inter_block_connections,
            self.bt_disulfide_conns,
            self.bt_atom_downstream_of_conn,
            self.global_params,
        )
