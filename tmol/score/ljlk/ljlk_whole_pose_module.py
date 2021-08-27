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
        super(self, LHLKWholePoseScoringModule).__init__()
        self.pose_stack_block_types = pose_stack_block_types
        self.pose_stack_min_block_bondsep = pose_stack_min_block_bondsep
        self.pose_stack_inter_block_bondsep = pose_stack_inter_block_bondsep
        self.bt_n_atoms = bt_n_atoms
        self.bt_n_heavy_atoms = bt_n_heavy_atoms
        self.bt_n_heavy_atoms_in_tile = bt_n_heavy_atoms_in_tile
        self.bt_heavy_atoms_in_tile = bt_heavy_atoms_in_tile
        self.bt_atom_types = bt_atom_types
        self.bt_heavy_atom_inds = bt_heavy_atom_inds
        self.bt_n_interblock_bonds = bt_n_interblock_bonds
        self.bt_atoms_forming_chemical_bonds = bt_atoms_forming_chemical_bonds
        self.bt_path_distance = bt_path_distance
        self.ljlk_type_params = ljlk_type_params
        self.global_params = global_params

    def forward(self, coords):
        return ljlk_pose_scores(
            coords,
            self.pose_stack_block_types,
            self.pose_min_block_bondsep,
            self.pose_inter_block_bondsep,
            self.bt_n_atoms,
            self.bt_n_heavy_atoms,
            self.bt_n_heavy_atoms_in_tile,
            self.bt_heavy_atoms_in_tile,
            self.bt_atom_types,
            self.bt_heavy_atom_inds,
            self.bt_n_interblock_bonds,
            self.bt_atoms_forming_chemical_bonds,
            self.bt_path_distance,
            self.ljlk_type_params,
            self.global_params,
        )
