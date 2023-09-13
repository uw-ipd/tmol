import torch


class RefWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_types,
        pose_stack_inter_block_connections,
        bt_atom_downstream_of_conn,
        ref_weights,
        # global_params,
    ):
        super(RefWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_inter_block_connections = _p(pose_stack_inter_block_connections)
        # self.bt_ref_quad_uaids = _p(bt_ref_quad_uaids)
        self.bt_atom_downstream_of_conn = _p(bt_atom_downstream_of_conn)
        self.ref_weights = _p(ref_weights)

        # self.global_params = _p(torch.stack(_t([global_params.K]), dim=1))

    def forward(self, coords):
        score = torch.index_select(
            self.ref_weights, 0, self.pose_stack_block_types.flatten()
        )
        score = torch.reshape(score, (self.pose_stack_block_types.size(0), -1))
        score = torch.sum(score, 1, keepdim=True)

        return score
