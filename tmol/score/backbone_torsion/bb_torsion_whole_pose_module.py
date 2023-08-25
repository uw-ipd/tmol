import torch

from tmol.score.backbone_torsion.potentials.compiled import backbone_torsion_pose_score


class BackboneTorsionWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_residue_connections,
        bt_atom_downstream_of_conn,
        bt_rama_table,
        bt_omega_table,
        bt_upper_conn_ind,
        bt_is_pro,
        bt_backbone_torsion_atoms,
        rama_tables,
        omega_tables,
        rama_table_params,
        omega_table_params,
    ):
        super(BackboneTorsionWholePoseScoringModule, self).__init__()

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
        self.bt_backbone_torsion_atoms = _p(bt_backbone_torsion_atoms)
        self.rama_tables = _p(rama_tables)
        self.omega_tables = _p(omega_tables)
        self.rama_table_params = _p(rama_table_params)
        self.omega_table_params = _p(omega_table_params)

    def forward(self, coords):
        return backbone_torsion_pose_score(
            coords,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_type,
            self.pose_stack_inter_residue_connections,
            self.bt_atom_downstream_of_conn,
            self.bt_rama_table,
            self.bt_upper_conn_ind,
            self.bt_is_pro,
            self.bt_backbone_torsion_atoms,
            self.rama_tables,
            self.omega_tables,
            self.rama_table_params,
            self.omega_table_params,
        )
