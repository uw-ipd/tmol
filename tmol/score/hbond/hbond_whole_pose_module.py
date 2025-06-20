import torch

from tmol.score.hbond.potentials.compiled import hbond_pose_scores
from tmol.score.common.convert_float64 import convert_float64


class HBondWholePoseScoringModule(torch.nn.Module):
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
        bt_tile_acc_inds,
        bt_tile_donor_type,
        bt_tile_acceptor_type,
        bt_tile_acceptor_hybridization,
        bt_atom_is_hydrogen,
        bt_path_distance,
        pair_params,
        pair_polynomials,
        global_params,
    ):
        super(HBondWholePoseScoringModule, self).__init__()

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

        self.bt_tile_acc_inds = _p(bt_tile_acc_inds)
        self.bt_tile_donor_type = _p(bt_tile_donor_type)
        self.bt_tile_acceptor_type = _p(bt_tile_acceptor_type)
        self.bt_tile_acceptor_hybridization = _p(bt_tile_acceptor_hybridization)
        self.bt_atom_is_hydrogen = _p(bt_atom_is_hydrogen)

        self.bt_path_distance = _p(bt_path_distance)
        self.pair_params = _p(pair_params)
        self.pair_polynomials = _p(pair_polynomials)
        self.global_params = _p(global_params)

    def forward(
        self, coords, block_pair_dispatch_indices, output_block_pair_energies=False
    ):
        args = [
            coords,
            block_pair_dispatch_indices,
            self.pose_stack_block_coord_offset,
            self.pose_stack_block_type,
            self.pose_stack_inter_residue_connections,
            self.pose_stack_min_bond_separation,
            self.pose_stack_inter_block_bondsep,
            self.bt_n_atoms,
            self.bt_n_interblock_bonds,
            self.bt_atoms_forming_chemical_bonds,
            self.bt_n_all_bonds,
            self.bt_all_bonds,
            self.bt_atom_all_bond_ranges,
            self.bt_tile_n_donH,
            self.bt_tile_n_acc,
            self.bt_tile_donH_inds,
            self.bt_tile_acc_inds,
            self.bt_tile_donor_type,
            self.bt_tile_acceptor_type,
            self.bt_tile_acceptor_hybridization,
            self.bt_atom_is_hydrogen,
            self.bt_path_distance,
            self.pair_params,
            self.pair_polynomials,
            self.global_params,
            output_block_pair_energies,
        ]

        if coords.dtype == torch.float64:
            convert_float64(args)

        return hbond_pose_scores(*args)
