import torch

from tmol.score.ljlk.potentials.compiled import ljlk_pose_scores


class LJLKWholePoseScoringModule(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_types,
        pose_stack_min_block_bondsep,
        pose_stack_inter_block_bondsep,
        bt_n_atoms,
        bt_n_heavy_atoms,
        bt_n_heavy_atoms_in_tile,
        bt_heavy_atoms_in_tile,
        bt_atom_types,
        bt_heavy_atom_inds,
        bt_n_interblock_bonds,
        bt_atoms_forming_chemical_bonds,
        bt_path_distance,
        ljlk_type_params,
        global_params,
    ):
        super(LJLKWholePoseScoringModule, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.pose_stack_block_types = _p(pose_stack_block_types)
        self.pose_stack_min_block_bondsep = _p(pose_stack_min_block_bondsep)
        self.pose_stack_inter_block_bondsep = _p(pose_stack_inter_block_bondsep)
        self.bt_n_atoms = _p(bt_n_atoms)
        self.bt_n_heavy_atoms_in_tile = _p(bt_n_heavy_atoms_in_tile)
        self.bt_heavy_atoms_in_tile = _p(bt_heavy_atoms_in_tile)
        self.bt_atom_types = _p(bt_atom_types)
        self.bt_heavy_atom_inds = _p(bt_heavy_atom_inds)
        self.bt_n_interblock_bonds = _p(bt_n_interblock_bonds)
        self.bt_atoms_forming_chemical_bonds = _p(bt_atoms_forming_chemical_bonds)
        self.bt_path_distance = _p(bt_path_distance)

        self.ljlk_type_params = _p(
            torch.stack(
                _t(
                    [
                        ljlk_type_params.lj_radius,
                        ljlk_type_params.lj_wdepth,
                        ljlk_type_params.lk_dgfree,
                        ljlk_type_params.lk_lambda,
                        ljlk_type_params.lk_volume,
                        ljlk_type_params.is_donor,
                        ljlk_type_params.is_hydroxyl,
                        ljlk_type_params.is_polarh,
                        ljlk_type_params.is_acceptor,
                    ]
                ),
                dim=1,
            )
        )

        self.global_params = _p(
            torch.stack(
                _t(
                    [
                        global_params.lj_hbond_dis,
                        global_params.lj_hbond_OH_donor_dis,
                        global_params.lj_hbond_hdis,
                    ]
                ),
                dim=1,
            )
        )

        # n_poses = pose_stack_block_types.shape[0]
        # max_n_blocks = pose_stack_block_types.shape[1]
        # device = pose_stack_block_types.device
        # self.scratch_block_spheres = _p(torch.zeros((n_poses, max_n_blocks, 4), dtype=torch.float32, device=device));
        # self.scratch_blocks_are_neighbors = _p(torch.zeros(n_poses, max_n_blocks, max_n_blocks), dtype=torch.int32, device=device)

        # self.ljlk_type_params = ljlk_type_params
        # self.global_params = global_params

    def forward(self, coords):
        return ljlk_pose_scores(
            coords,
            self.pose_stack_block_types,
            self.pose_stack_min_block_bondsep,
            self.pose_stack_inter_block_bondsep,
            self.bt_n_atoms,
            self.bt_n_heavy_atoms_in_tile,
            self.bt_heavy_atoms_in_tile,
            self.bt_atom_types,
            self.bt_n_interblock_bonds,
            self.bt_atoms_forming_chemical_bonds,
            self.bt_path_distance,
            self.ljlk_type_params,
            self.global_params,
        )
