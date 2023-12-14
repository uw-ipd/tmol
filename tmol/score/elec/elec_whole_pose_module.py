import torch

from tmol.score.elec.potentials.compiled import elec_pose_scores


class ElecWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_types,
        pose_stack_min_block_bondsep,
        pose_stack_inter_block_bondsep,
        bt_n_atoms,
        bt_partial_charge,
        bt_n_interblock_bonds,
        bt_atoms_forming_chemical_bonds,
        bt_inter_repr_path_distance,
        bt_intra_repr_path_distance,
        global_params,
    ):
        super(ElecWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_min_block_bondsep = _p(pose_stack_min_block_bondsep)
        self.pose_stack_inter_block_bondsep = _p(pose_stack_inter_block_bondsep)
        self.bt_n_atoms = _p(bt_n_atoms)
        self.bt_partial_charge = _p(bt_partial_charge)
        self.bt_n_interblock_bonds = _p(bt_n_interblock_bonds)
        self.bt_atoms_forming_chemical_bonds = _p(bt_atoms_forming_chemical_bonds)
        self.bt_inter_repr_path_distance = _p(bt_inter_repr_path_distance)
        self.bt_intra_repr_path_distance = _p(bt_intra_repr_path_distance)

        self.global_params = _p(
            torch.tensor(
                [
                    global_params.elec_sigmoidal_die_D,
                    global_params.elec_sigmoidal_die_D0,
                    global_params.elec_sigmoidal_die_S,
                    global_params.elec_min_dis,
                    global_params.elec_max_dis,
                ],
                dtype=torch.float32,
                device=pose_stack_block_coord_offset.device,
            )[None, :]
        )

    def forward(self, coords, output_block_pair_energies=False):
        return elec_pose_scores(
            coords,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_types,
            self.pose_stack_min_block_bondsep,
            self.pose_stack_inter_block_bondsep,
            self.bt_n_atoms,
            self.bt_partial_charge,
            self.bt_n_interblock_bonds,
            self.bt_atoms_forming_chemical_bonds,
            self.bt_inter_repr_path_distance,
            self.bt_intra_repr_path_distance,
            self.global_params,
            output_block_pair_energies,
        )
