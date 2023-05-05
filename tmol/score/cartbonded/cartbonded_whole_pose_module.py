import torch

from tmol.score.cartbonded.potentials.compiled import cartbonded_pose_scores


class CartBondedWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_types,
        pose_stack_inter_block_connections,
        atom_paths_from_conn,
        atom_unique_ids,
        hash_keys,
        hash_values,
        cart_subgraphs,
        cart_subgraph_offsets
        # global_params,
    ):
        super(CartBondedWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_inter_block_connections = _p(pose_stack_inter_block_connections)
        self.atom_paths_from_conn = _p(atom_paths_from_conn)
        self.atom_unique_ids = _p(atom_unique_ids)
        self.hash_keys = _p(hash_keys)
        self.hash_values = _p(hash_values)
        self.cart_subgraphs = _p(cart_subgraphs)
        self.cart_subgraph_offsets = _p(cart_subgraph_offsets)

        # self.global_params = _p(torch.stack(_t([global_params.K]), dim=1))

    def forward(self, coords):
        return cartbonded_pose_scores(
            coords,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_types,
            self.pose_stack_inter_block_connections,
            self.atom_paths_from_conn,
            self.atom_unique_ids,
            self.hash_keys,
            self.hash_values,
            self.cart_subgraphs,
            self.cart_subgraph_offsets
            # self.global_params,
        )
