import attr

import torch

from tmol.score.dunbrack.potentials.compiled import dunbrack_pose_scores

import dataclasses


class DunbrackWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_types,
        pose_stack_inter_block_connections,
        bt_atom_downstream_of_conn,
        global_params,
        dunbrack_packed_block_data,
    ):
        super(DunbrackWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_inter_block_connections = _p(pose_stack_inter_block_connections)
        self.bt_atom_downstream_of_conn = _p(bt_atom_downstream_of_conn)

        self.dunbrack_database = [_p(f) for f in global_params]

        self.dunbrack_packed_block_data = [_p(f) for f in dunbrack_packed_block_data]

    def forward(self, coords):
        return dunbrack_pose_scores(
            coords,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_types,
            self.pose_stack_inter_block_connections,
            self.bt_atom_downstream_of_conn,
            *self.dunbrack_database,
            *self.dunbrack_packed_block_data,
        )
