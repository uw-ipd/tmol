import torch


class RefWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_types,
        pose_stack_inter_block_connections,
        bt_atom_downstream_of_conn,
        ref_weights,
    ):
        super(RefWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_inter_block_connections = _p(pose_stack_inter_block_connections)
        self.bt_atom_downstream_of_conn = _p(bt_atom_downstream_of_conn)
        self.ref_weights = _p(ref_weights)

    def forward(self, coords):
        block_types = self.pose_stack_block_types

        # fill our per-block ref scores with zeros to start
        score = torch.zeros_like(block_types, dtype=torch.float32)

        # grab the indices of any non-negative (real) blocks
        real_blocks = block_types >= 0

        # fill out the scores for the real blocks by dereferencing the block types into the ref weights
        score[real_blocks] = torch.index_select(
            self.ref_weights, 0, block_types[real_blocks]
        )

        # for each pose, sum up the block scores
        score = torch.sum(score, 1)

        # wrap this all in an extra dim (the output expects an outer dim to separate sub-terms)
        score = torch.unsqueeze(score, 0)

        score.requires_grad = True  # a bit of a hack to make the benchmark test not error out because there are no grads

        return score
