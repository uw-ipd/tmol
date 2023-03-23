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

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_inter_block_connections = _p(pose_stack_inter_block_connections)
        self.bt_disulfide_conns = _p(bt_disulfide_conns)
        self.bt_atom_downstream_of_conn = _p(bt_atom_downstream_of_conn)

        self.global_params = _p(
            torch.stack(
                _t(
                    [
                        global_params.d_location,
                        global_params.d_scale,
                        global_params.d_shape,
                        global_params.a_logA,
                        global_params.a_kappa,
                        global_params.a_mu,
                        global_params.dss_logA1,
                        global_params.dss_kappa1,
                        global_params.dss_mu1,
                        global_params.dss_logA2,
                        global_params.dss_kappa2,
                        global_params.dss_mu2,
                        global_params.dcs_logA1,
                        global_params.dcs_mu1,
                        global_params.dcs_kappa1,
                        global_params.dcs_logA2,
                        global_params.dcs_mu2,
                        global_params.dcs_kappa2,
                        global_params.dcs_logA3,
                        global_params.dcs_mu3,
                        global_params.dcs_kappa3,
                    ]
                ),
                dim=1,
            )
        )

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
