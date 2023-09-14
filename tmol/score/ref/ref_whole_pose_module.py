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
        score = torch.add(
            self.pose_stack_block_types, 1
        ).flatten()  # add 1 to the tensor to handle -1 block types (they will now index 0, which will score as 0 in the weights table). Also flatten it so that we can do the operation on all blocks/poses at the same time.
        score = torch.index_select(
            self.ref_weights, 0, score
        )  # for all blocks in all poses, do a lookup of that block type in the weights table and use that value instead.
        score = torch.reshape(
            score, (self.pose_stack_block_types.size(0), -1)
        )  # separate our blocks back into their appropriate poses
        score = torch.sum(score, 1)  # sum the block scores for each pose
        score = torch.unsqueeze(score, 0)  # add back in the outermose dimension

        return score
