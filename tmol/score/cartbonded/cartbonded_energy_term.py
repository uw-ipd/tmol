import torch
import numpy

from itertools import permutations

from tmol.score.atom_type_dependent_term import AtomTypeDependentTerm

from tmol.database import ParameterDatabase

# from tmol.score.cartbonded.params import CartBondedGlobalParams
from tmol.score.cartbonded.cartbonded_whole_pose_module import (
    CartBondedWholePoseScoringModule,
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
        self.device = device

    @classmethod
    def score_types(cls):
        import tmol.score.terms.cartbonded_creator

        return tmol.score.terms.cartbonded_creator.CartBondedTermCreator.score_types()

    def n_bodies(self):
        return 1

    def find_subgraphs(self, bonds, block_type):  # noqa: C901
        lengths = []
        angles = []
        torsions = []
        improper = []

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

        # get improper torsions
        for atom3 in [block_type.atom_to_idx["CA"]]:
            comb = list(permutations(bondmap[atom3], 3))
            for atom1, atom2, atom4 in comb:
                improper.append((atom1, atom2, atom3, atom4))

        return (
            lengths,
            angles,
            torsions,
            improper,
        )

    def get_raw_params_for_res(self, res):
        return (
            self.cart_database.residue_params[res].length_parameters
            + self.cart_database.residue_params[res].angle_parameters
            + self.cart_database.residue_params[res].torsion_parameters
            + self.cart_database.residue_params[res].improper_parameters
            + self.cart_database.residue_params[res].hxltorsion_parameters
        )

    def get_formatted_atoms_and_params(self, raw_params):
        fields = ["atm1", "atm2", "atm3", "atm4"]
        atoms = [
            getattr(raw_params, field) for field in fields if hasattr(raw_params, field)
        ]
        fields = ["type", "x0", "K", "k1", "k2", "k3", "phi1", "phi2", "phi3"]
        params = [
            getattr(raw_params, field) for field in fields if hasattr(raw_params, field)
        ]
        return atoms, params

    def get_params_for_res(self, res):
        params_by_atom_unique_id = {}

        # Fetch the raw params from the DB
        all_params = self.get_raw_params_for_res(res)

        for param in all_params:
            # Format the raw param
            atoms, params = self.get_formatted_atoms_and_params(param)

            previous_atm = ""
            is_wildcard = False
            for i, atom in enumerate(atoms):
                # Check if these atoms should be wildcard ids. This includes atoms after any bonds that span the residue connection.
                if (
                    res == "wildcard"
                    or not (hasattr(param, "type") and param.type == 3)
                    and (
                        previous_atm == "N"
                        and atom == "C"
                        or previous_atm == "C"
                        and atom == "N"
                    )
                ):
                    is_wildcard = True

                previous_atm = atom
                atoms[i] = (
                    self.get_atom_wildcard_id_name(res, atom)
                    if is_wildcard
                    else self.get_atom_unique_id_name(res, atom)
                )
                # If the atom isn't recorded in the atom_unique_id_index (inherited from AtomTypeDependentTerm), add it
                if atoms[i] not in self.atom_unique_id_index:
                    self.atom_unique_id_index[atoms[i]] = len(self.atom_unique_id_index)

            key = tuple([self.atom_unique_id_index[atom] for atom in atoms])

            params_by_atom_unique_id[key] = params

        return params_by_atom_unique_id

    def setup_block_type(self, block_type: RefinedResidueType):
        super(CartBondedEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "cartbonded_subgraphs"):
            assert hasattr(block_type, "cartbonded_params")
            return

        # Get the subgraphs for this block type
        lengths, angles, torsions, improper = self.find_subgraphs(
            block_type.bond_indices, block_type
        )
        cart_subgraphs = numpy.asarray(lengths + angles + torsions + improper)
        setattr(block_type, "cartbonded_subgraphs", cart_subgraphs)

        # Fetch the params from the database, updating the atom id store if necessary
        cartbonded_params = self.get_params_for_res(block_type.name)
        setattr(block_type, "cartbonded_params", cartbonded_params)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(CartBondedEnergyTerm, self).setup_packed_block_types(packed_block_types)
        if hasattr(packed_block_types, "cartbonded_subgraphs"):
            assert hasattr(packed_block_types, "cartbonded_subgraph_offsets")
            assert hasattr(packed_block_types, "cartbonded_max_subgraphs_per_block")
            assert hasattr(packed_block_types, "cartbonded_params_hash_keys")
            assert hasattr(packed_block_types, "cartbonded_params_hash_values")
            return

        # Aggregate the subgraphs and collect metadata
        total_subgraphs = sum(
            bt.cartbonded_subgraphs.shape[0]
            for bt in packed_block_types.active_block_types
        )
        subgraphs = numpy.full((total_subgraphs, 4), -1, dtype=numpy.int32)
        subgraph_offsets = []
        offset = 0
        max_subgraphs_per_block = 0
        for block_type in packed_block_types.active_block_types:
            subgraph_offsets.append(offset)

            n_subgraphs = block_type.cartbonded_subgraphs.shape[0]
            subgraphs[offset : offset + n_subgraphs] = block_type.cartbonded_subgraphs
            offset += n_subgraphs

            max_subgraphs_per_block = max(
                max_subgraphs_per_block, offset - subgraph_offsets[-1]
            )
        subgraphs = torch.from_numpy(subgraphs).to(device=self.device)
        subgraph_offsets = numpy.asarray(subgraph_offsets, dtype=numpy.int32)
        subgraph_offsets = torch.from_numpy(subgraph_offsets).to(device=self.device)
        setattr(packed_block_types, "cartbonded_subgraphs", subgraphs)
        setattr(packed_block_types, "cartbonded_subgraph_offsets", subgraph_offsets)
        setattr(
            packed_block_types,
            "cartbonded_max_subgraphs_per_block",
            max_subgraphs_per_block,
        )

        # Aggregate the params

        # Calculate the total number of params
        total_params = sum(
            [len(bt.cartbonded_params) for bt in packed_block_types.active_block_types]
        )
        # get the params not associated with any specific residue
        wildcard_params = self.get_params_for_res("wildcard").items()
        total_params += len(wildcard_params)

        # The hash function for storing the parameters in the tensors across the python/c++ barier.
        # Note that the implementation of this function MUST match the implementation defined in the
        # C++.
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

        # Construct the params hash with the given scaling factor
        SCALE = 2
        hash_keys = numpy.full(
            (total_params * SCALE, 5),
            -1,
            dtype=numpy.int32,
        )
        hash_values = numpy.full(
            (total_params, 7),
            0,
            dtype=numpy.float32,
        )

        cur_val = 0
        for bt in packed_block_types.active_block_types:
            for key, value in bt.cartbonded_params.items():
                add_to_hash(hash_keys, hash_values, cur_val, key, value)
                cur_val += 1

        for key, value in wildcard_params:
            add_to_hash(hash_keys, hash_values, cur_val, key, value)
            cur_val += 1

        hash_keys_tensor = torch.from_numpy(hash_keys).to(device=self.device)
        hash_values_tensor = torch.from_numpy(hash_values).to(device=self.device)
        setattr(packed_block_types, "cartbonded_params_hash_keys", hash_keys_tensor)
        setattr(packed_block_types, "cartbonded_params_hash_values", hash_values_tensor)

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
            atom_wildcard_ids=pbt.atom_wildcard_ids,
            hash_keys=pbt.cartbonded_params_hash_keys,
            hash_values=pbt.cartbonded_params_hash_values,
            cart_subgraphs=pbt.cartbonded_subgraphs,
            cart_subgraph_offsets=pbt.cartbonded_subgraph_offsets,
            max_subgraphs_per_block=pbt.cartbonded_max_subgraphs_per_block,
        )
