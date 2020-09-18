import attr
import torch

from ..AtomTypeDependentTerm import AtomTypeDependentTerm
from ..BondDependentTerm import BondDependentTerm
from .params import LJLKTypeParams, LJLKGlobalParams

from tmol.system.pose import PackedBlockTypes


@attr.s(auto_attribs=True)
class LJLKEnergy(AtomTypeDependentTerm, BondDependentTerm):
    type_params: LJLKTypeParams
    global_params: LJLKGlobalParams

    def setup_packed_block_types(self, packed_block_types: PackedBlockypes):
        super(LJLKEnergy, self).setup_packed_block_types(packed_block_types)

    def setup_poses(self, poses: Poses):
        super(LJLKEnergy, self).setup_poses(poses)

    def inter_module(
        self,
        packed_block_types: PackedBlockTypes,
        systems: Poses,
        context_system_ids: Tensor(int)[:, :],
        system_bounding_spheres: Tensor(float)[:, :, 4],
    ):
        system_neighbor_list = create_block_neighbor_lists(
            systems, system_bounding_spheres
        )

        return LJLKInterSystemModule(
            context_system_ids=self.context_system_ids,
            system_min_bond_separation=systems.min_bond_separation,
            system_interblock_bonds=interblock_bonds,
            system_neighbor_list=system_neighbor_list,
            block_type_n_atoms=packed_block_types.block_type_n_atoms,
            block_type_atom_types=packed_block_types.atom_types,
            block_type_n_interblock_bonds=n_interblock_bonds,
            block_type_atoms_forming_chemical_bonds=packed_block_types.atoms_forming_chemical_bonds,
            block_type_path_distance=packed_block_types.path_distance,
            type_params=self.type_params,
            global_params=self.global_params,
        )


class LJLKInterSystemModule(torch.jit.ScriptModule):
    def __init__(
        self,
        context_system_ids,
        system_min_bond_separation,
        system_interblock_bonds,
        system_neighbor_list,
        block_type_n_atoms,
        block_type_atom_types,
        block_type_n_interblock_bonds,
        block_type_atoms_forming_chemical_bonds,
        block_type_path_distance,
        type_params,
        global_params,
    ):
        self.context_system_ids = context_system_ids
        self.system_min_bond_separation = system_min_bond_separation
        self.system_interblock_bonds = system_interblock_bonds
        self.system_neighbor_list = system_neighbor_list
        self.block_type_n_atoms = block_type_n_atoms
        self.block_type_atom_types = block_type_atom_types
        self.block_type_n_interblock_bonds = block_type_n_interblock_bonds
        self.block_type_atoms_forming_chemical_bonds = (
            block_type_atoms_forming_chemical_bonds
        )
        self.block_type_path_distance = block_type_path_distance
        self.type_params = type_params
        self.global_params = global_params

    @torch.jit.script_method
    def forward(
        self, context_coords, context_block_type, alternate_coords, alternate_ids
    ):
        return torch.ops.tmol.score_ljlk_inter_system_scores(
            context_coords,
            context_block_type,
            alternate_coords,
            alternate_ids,
            self.context_system_ids,
            self.system_min_bond_separation,
            self.system_interblock_bonds,
            self.system_neighbor_list,
            self.block_type_n_atoms,
            self.block_type_atom_types,
            self.block_type_n_interblock_bonds,
            self.block_type_atoms_forming_chemical_bonds,
            self.block_type_path_distance,
            self.type_params,
            self.global_params,
        )
