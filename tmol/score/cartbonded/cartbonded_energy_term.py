import torch
import numpy

from tmol.score.atom_type_dependent_term import AtomTypeDependentTerm

from tmol.database import ParameterDatabase

# from tmol.score.cartbonded.params import CartBondedGlobalParams
from tmol.score.cartbonded.cartbonded_whole_pose_module import (
    CartBondedWholePoseScoringModule
)

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

debug = False


class CartBondedEnergyTerm(AtomTypeDependentTerm):
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(CartBondedEnergyTerm, self).__init__(param_db=param_db, device=device)

        self.cart_database = param_db.scoring.cartbonded
        """self.global_params = CartBondedGlobalParams.from_database(
            param_db.scoring.cartbonded, device
        )"""
        self.device = device

    @classmethod
    def score_types(cls):
        import tmol.score.terms.omega_creator

        return tmol.score.terms.omega_creator.CartBondedTermCreator.score_types()

    def n_bodies(self):
        return 1

    def find_subgraphs(self, bonds):
        lengths = []
        angles = []
        torsions = []

        # create a convenient datastructure for following connections
        bondmap = {}
        for bond in bonds:
            if bond[0] not in bondmap.keys():
                bondmap[bond[0]] = set()
            bondmap[bond[0]].add(bond[1])

        # get lengths
        for atom1 in bondmap.keys():
            for atom2 in bondmap[atom1]:
                if atom1 < atom2:
                    lengths.append((atom1, atom2, -1, -1))

        # get angles
        for atom1 in bondmap.keys():
            for atom2 in bondmap[atom1]:
                for atom3 in bondmap[atom2]:
                    if atom1 >= atom3:
                        continue
                    angles.append((atom1, atom2, atom3, -1))

        # get torsions
        for atom1 in bondmap.keys():
            for atom2 in bondmap[atom1]:
                for atom3 in bondmap[atom2]:
                    if atom1 == atom3:
                        continue
                    for atom4 in bondmap[atom3]:
                        if atom2 == atom4:
                            continue
                        if atom1 >= atom4:
                            continue
                        torsions.append((atom1, atom2, atom3, atom4))

        if debug:
            print("")
            print("BONDS")
            print(bonds)
            print("")
            print("LENGTHS/ANGLES/TORSIONS")
            print(lengths)
            print(angles)
            print(torsions)

        return numpy.asarray(lengths), numpy.asarray(angles), numpy.asarray(torsions)

    def setup_block_type(self, block_type: RefinedResidueType):
        super(CartBondedEnergyTerm, self).setup_block_type(block_type)

        lengths, angles, torsions = self.find_subgraphs(block_type.bond_indices)

        setattr(block_type, "cart_lengths", lengths)
        setattr(block_type, "cart_angles", angles)
        setattr(block_type, "cart_torsions", torsions)

        params_by_atom_unique_id = {}
        all_params = (
            self.cart_database.length_parameters
            + self.cart_database.angle_parameters
            + self.cart_database.torsion_parameters
        )
        for param in all_params:
            if param.res != block_type.name:
                continue

            fields = ["atm1", "atm2", "atm3", "atm4"]
            atoms = [getattr(param, field) for field in fields if hasattr(param, field)]
            fields = ["x0", "K", "period"]
            params = [
                getattr(param, field) for field in fields if hasattr(param, field)
            ]

            for i, atom in enumerate(atoms):
                is_wildcard = False
                atoms[i] = (
                    self.get_atom_wildcard_id_name(param.res, atom)
                    if is_wildcard
                    else self.get_atom_unique_id_name(param.res, atom)
                )
                if atoms[i] not in self.atom_unique_id_index:
                    self.atom_unique_id_index[atoms[i]] = len(self.atom_unique_id_index)

            key = tuple([self.atom_unique_id_index[atom] for atom in atoms])

            params_by_atom_unique_id[key] = params

        if debug:
            print("")
            print("PARAMS")
            print(params_by_atom_unique_id)
        setattr(block_type, "params_by_unique_id", params_by_atom_unique_id)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(CartBondedEnergyTerm, self).setup_packed_block_types(packed_block_types)

        # collect lengths/angles/torsions
        total_subgraphs = 0
        for block_type in packed_block_types.active_block_types:
            total_subgraphs += block_type.cart_lengths.shape[0]
            total_subgraphs += block_type.cart_angles.shape[0]
            total_subgraphs += block_type.cart_torsions.shape[0]
        subgraphs = numpy.full((total_subgraphs, 4), -1, dtype=numpy.int32)
        subgraph_offsets = []
        offset = 0
        for i, block_type in enumerate(packed_block_types.active_block_types):
            subgraph_offsets.append(offset)

            lengths = block_type.cart_lengths.shape[0]
            subgraphs[offset : offset + lengths] = block_type.cart_lengths
            offset += lengths

            angles = block_type.cart_angles.shape[0]
            subgraphs[offset : offset + angles] = block_type.cart_angles
            offset += angles

            torsions = block_type.cart_torsions.shape[0]
            subgraphs[offset : offset + torsions] = block_type.cart_torsions
            offset += torsions

        if debug:
            print(subgraphs)
            print(subgraph_offsets)
        subgraph_offsets = numpy.asarray(subgraph_offsets, dtype=numpy.int32)
        subgraphs = torch.from_numpy(subgraphs).to(device=self.device)
        subgraph_offsets = torch.from_numpy(subgraph_offsets).to(device=self.device)
        setattr(packed_block_types, "cart_subgraphs", subgraphs)
        setattr(packed_block_types, "cart_subgraph_offsets", subgraph_offsets)

        total_params = sum(
            [
                len(bt.params_by_unique_id)
                for bt in packed_block_types.active_block_types
            ]
        )
        scale = 2

        # stupid hash function
        def hash_fun(key, max_size):
            value = 0x1234
            for k in key:
                value = (k ^ value) * 3141 % max_size  # XOR
            return value

        def add_to_hash(hash_keys, hash_values, max_value_index, key, values):
            index = hash_fun(key, hash_keys.shape[0])
            while hash_keys[index][0] != -1:
                index = (index + 1) % hash_keys.shape[0]
            for i, k in enumerate(key):
                hash_keys[index][i] = k
            hash_keys[index][4] = max_value_index

            for i, value in enumerate(values):
                hash_values[max_value_index][i] = value

        hash_keys = numpy.full(
            (total_params * scale, 5),  # atom1, atom2, atom3, atom4, index
            -1,
            dtype=numpy.int32,
        )

        hash_values = numpy.full(
            (total_params, 3), -1, dtype=numpy.float32  # x0, K, period
        )

        cur_val = 0
        for bt in packed_block_types.active_block_types:
            for key, value in bt.params_by_unique_id.items():
                add_to_hash(hash_keys, hash_values, cur_val, key, value)
                cur_val += 1

        hash_keys_tensor = torch.from_numpy(hash_keys).to(device=self.device)
        hash_values_tensor = torch.from_numpy(hash_values).to(device=self.device)
        setattr(packed_block_types, "hash_keys", hash_keys_tensor)
        setattr(packed_block_types, "hash_values", hash_values_tensor)

    def setup_poses(self, poses: PoseStack):
        super(CartBondedEnergyTerm, self).setup_poses(poses)

    def render_whole_pose_scoring_module(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return CartBondedWholePoseScoringModule(
            pose_stack_block_coord_offset=pose_stack.block_coord_offset,
            pose_stack_block_types=pose_stack.block_type_ind,
            pose_stack_inter_block_connections=pose_stack.inter_residue_connections,
            atom_paths_from_conn=pbt.atom_paths_from_conn,
            atom_unique_ids=pbt.atom_unique_ids,
            hash_keys=pbt.hash_keys,
            hash_values=pbt.hash_values,
            cart_subgraphs=pbt.cart_subgraphs,
            cart_subgraph_offsets=pbt.cart_subgraph_offsets
            # global_params=self.global_params,
        )
