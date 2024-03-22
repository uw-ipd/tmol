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
        atom_wildcard_ids,
        hash_keys,
        hash_values,
        cart_subgraphs,
        cart_subgraph_offsets,
        max_subgraphs_per_block,
    ):
        super(CartBondedWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_inter_block_connections = _p(pose_stack_inter_block_connections)
        self.atom_paths_from_conn = _p(atom_paths_from_conn)
        self.atom_unique_ids = _p(atom_unique_ids)
        self.atom_wildcard_ids = _p(atom_wildcard_ids)
        self.hash_keys = _p(hash_keys)
        self.hash_values = _p(hash_values)
        self.cart_subgraphs = _p(cart_subgraphs)
        self.cart_subgraph_offsets = _p(cart_subgraph_offsets)
        self.max_subgraphs_per_block = torch.tensor(max_subgraphs_per_block)

    def forward(self, coords, output_block_pair_energies=False):
        return cartbonded_pose_scores(
            coords,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_types,
            self.pose_stack_inter_block_connections,
            self.atom_paths_from_conn,
            self.atom_unique_ids,
            self.atom_wildcard_ids,
            self.hash_keys,
            self.hash_values,
            self.cart_subgraphs,
            self.cart_subgraph_offsets,
            self.max_subgraphs_per_block,
            output_block_pair_energies,
        )
