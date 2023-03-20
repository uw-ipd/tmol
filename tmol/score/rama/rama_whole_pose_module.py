import torch

from tmol.score.rama.potentials.compiled import pose_score_rama


class RamaWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_residue_connections,
        bt_atom_downstream_of_conn,
        bt_rama_table,
        bt_upper_conn_ind,
        bt_is_pro,
        bt_rama_torsion_atoms,
        rama_tables,
        table_params,
    ):
        super(RamaWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_type = _p(pose_stack_block_type)
        self.pose_stack_inter_residue_connections = _p(
            pose_stack_inter_residue_connections
        )
        self.bt_atom_downstream_of_conn = _p(bt_atom_downstream_of_conn)
        self.bt_rama_table = _p(bt_rama_table)
        self.bt_upper_conn_ind = _p(bt_upper_conn_ind)
        self.bt_is_pro = _p(bt_is_pro)
        self.bt_rama_torsion_atoms = _p(bt_rama_torsion_atoms)
        self.rama_tables = _p(rama_tables)
        self.table_params = _p(table_params)

    def forward(self, coords):
        """Two step scoring: first build the waters and then score;
        derivatives are calculated backwards through the water
        building step by torch's autograd machinery
        """

        return pose_score_rama(
            coords,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_type,
            self.pose_stack_inter_residue_connections,
            self.bt_atom_downstream_of_conn,
            self.bt_rama_table,
            self.bt_upper_conn_ind,
            self.bt_is_pro,
            self.bt_rama_torsion_atoms,
            self.rama_tables,
            self.table_params,
        )
