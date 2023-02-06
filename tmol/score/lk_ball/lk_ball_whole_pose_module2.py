import torch

from tmol.score.lk_ball.potentials.compiled import gen_pose_waters, pose_score_lk_ball2


class LKBallWholePoseScoringModule2(torch.nn.Module):
    def __init__(
        self,
        pose_stack_block_coord_offset,
        pose_stack_block_type,
        pose_stack_inter_residue_connections,
        pose_stack_min_bond_separation,
        pose_stack_inter_block_bondsep,
        bt_n_atoms,
        bt_n_interblock_bonds,
        bt_atoms_forming_chemical_bonds,
        bt_n_all_bonds,
        bt_all_bonds,
        bt_atom_all_bond_ranges,
        bt_tile_n_donH,
        bt_tile_n_acc,
        bt_tile_donH_inds,
        bt_tile_don_hvy_inds,
        bt_tile_which_donH_for_hvy,
        bt_tile_acc_inds,
        bt_tile_hybridization,
        bt_tile_acc_n_attached_H,
        bt_atom_is_hydrogen,
        bt_tile_n_polar_atoms,
        bt_tile_n_occluder_atoms,
        bt_tile_pol_occ_inds,
        bt_tile_pol_occ_n_waters,
        bt_tile_lk_ball_params,
        bt_path_distance,
        lk_ball_global_params,
        water_gen_global_params,
        sp2_water_tors,
        sp3_water_tors,
        ring_water_tors,
    ):
        super(LKBallWholePoseScoringModule2, self).__init__()

        def _p(t):
            return torch.nn.Parameter(t, requires_grad=False)

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        self.pose_stack_block_coord_offset = _p(pose_stack_block_coord_offset)
        self.pose_stack_block_type = _p(pose_stack_block_type)
        self.pose_stack_inter_residue_connections = _p(
            pose_stack_inter_residue_connections
        )
        self.pose_stack_min_bond_separation = _p(pose_stack_min_bond_separation)
        self.pose_stack_inter_block_bondsep = _p(pose_stack_inter_block_bondsep)

        self.bt_n_atoms = _p(bt_n_atoms)
        self.bt_n_interblock_bonds = _p(bt_n_interblock_bonds)
        self.bt_atoms_forming_chemical_bonds = _p(bt_atoms_forming_chemical_bonds)
        self.bt_n_all_bonds = _p(bt_n_all_bonds)
        self.bt_all_bonds = _p(bt_all_bonds)

        self.bt_atom_all_bond_ranges = _p(bt_atom_all_bond_ranges)
        self.bt_tile_n_donH = _p(bt_tile_n_donH)
        self.bt_tile_n_acc = _p(bt_tile_n_acc)
        self.bt_tile_donH_inds = _p(bt_tile_donH_inds)
        self.bt_tile_don_hvy_inds = _p(bt_tile_don_hvy_inds)

        self.bt_tile_which_donH_for_hvy = _p(bt_tile_which_donH_for_hvy)
        self.bt_tile_acc_inds = _p(bt_tile_acc_inds)
        self.bt_tile_hybridization = _p(bt_tile_hybridization)
        self.bt_tile_acc_n_attached_H = _p(bt_tile_acc_n_attached_H)
        self.bt_atom_is_hydrogen = _p(bt_atom_is_hydrogen)

        self.bt_tile_n_polar_atoms = _p(bt_tile_n_polar_atoms)
        self.bt_tile_n_occluder_atoms = _p(bt_tile_n_occluder_atoms)
        self.bt_tile_pol_occ_inds = _p(bt_tile_pol_occ_inds)
        self.bt_tile_pol_occ_n_waters = _p(bt_tile_pol_occ_n_waters)
        self.bt_tile_lk_ball_params = _p(bt_tile_lk_ball_params)

        self.bt_path_distance = _p(bt_path_distance)
        self.lk_ball_global_params = _p(lk_ball_global_params)
        self.water_gen_global_params = _p(water_gen_global_params)
        self.sp2_water_tors = _p(sp2_water_tors)
        self.sp3_water_tors = _p(sp3_water_tors)

        self.ring_water_tors = _p(ring_water_tors)

    def forward(self, pose_coords):
        """Two step scoring: first build the waters and then score;
        derivatives are calculated backwards through the water
        building step by torch's autograd machinery
        """

        water_coords = gen_pose_waters(
            pose_coords,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_type,
            self.pose_stack_inter_residue_connections,
            self.bt_n_atoms,
            self.bt_n_interblock_bonds,
            self.bt_atoms_forming_chemical_bonds,
            self.bt_n_all_bonds,
            self.bt_all_bonds,
            self.bt_atom_all_bond_ranges,
            self.bt_tile_n_donH,
            self.bt_tile_n_acc,
            self.bt_tile_donH_inds,
            self.bt_tile_don_hvy_inds,
            self.bt_tile_which_donH_for_hvy,
            self.bt_tile_acc_inds,
            self.bt_tile_hybridization,
            self.bt_tile_acc_n_attached_H,
            self.bt_atom_is_hydrogen,
            self.water_gen_global_params,
            self.sp2_water_tors,
            self.sp3_water_tors,
            self.ring_water_tors,
        )

        return pose_score_lk_ball2(
            pose_coords,
            water_coords,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_type,
            self.pose_stack_inter_residue_connections,
            self.pose_stack_min_bond_separation,
            self.pose_stack_inter_block_bondsep,
            self.bt_n_atoms,
            self.bt_n_interblock_bonds,
            self.bt_atoms_forming_chemical_bonds,
            self.bt_tile_n_polar_atoms,
            self.bt_tile_n_occluder_atoms,
            self.bt_tile_pol_occ_inds,
            self.bt_tile_pol_occ_n_waters,
            self.bt_tile_lk_ball_params,
            self.bt_path_distance,
            self.lk_ball_global_params,
        )
