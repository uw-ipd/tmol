import torch
import numpy
import attrs

from itertools import permutations

from tmol.score import AtomTypeDependentTerm

from tmol.database import ParameterDatabase

from tmol.chemical import RefinedResidueType
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)
from tmol.score.common import make_hashtable_keys_values, add_to_hashtable

debug = False

# marks an atom in a cartbonded.yaml param as being on the far side of a
# residue connection
CROSS_RES_PREFIX = "+"


@attrs.define(auto_attribs=True, slots=True, frozen=True)
class CartBondedBlockAnnotations:
    cartbonded_subgraphs: torch.Tensor
    cartbonded_subgraph_type_counts: torch.Tensor
    cartbonded_subgraph_type_offsets: torch.Tensor
    cartbonded_params: dict


@attrs.define(auto_attribs=True, slots=True, frozen=True)
class CartBondedPackedBlockTypesAnnotations:
    cartbonded_subgraphs: torch.Tensor
    cartbonded_subgraph_offsets: torch.Tensor
    cartbonded_subgraph_type_counts: torch.Tensor
    cartbonded_subgraph_type_offsets: torch.Tensor
    cartbonded_subgraph_param_indices: torch.Tensor
    cartbonded_max_subgraphs_per_block: int
    cartbonded_atom_unique_id_index: dict
    cartbonded_params_hash_keys: torch.Tensor
    cartbonded_params_hash_values: torch.Tensor


class CartBondedEnergyTerm(AtomTypeDependentTerm):
    device: torch.device  # = attr.ib()
    improper_roots: set()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(CartBondedEnergyTerm, self).__init__(param_db=param_db, device=device)

        # Find the root of the improper torsions so that we can annotate them in the block types
        def find_improper_roots(db):
            roots = set()
            for res, params in db.residue_params.items():
                for imp in params.improper_parameters:
                    roots.add(imp.atm3.lstrip(CROSS_RES_PREFIX))
            return roots

        self.improper_roots = find_improper_roots(param_db.scoring.cartbonded)

        self.cart_database = param_db.scoring.cartbonded
        self.hash = self.cart_database.hash
        self.device = device
        self._params_for_res_cache = {}

    @classmethod
    def class_name(cls):
        return "CartBonded"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._cartbonded_creator

        return tmol.score.terms._cartbonded_creator.CartBondedTermCreator.score_types()

    def n_bodies(self):
        return 2

    def find_subgraphs(self, bonds, block_type):  # noqa: C901
        lengths = []
        angles = []
        torsions = []
        improper = []

        # create a convenient datastructure for following connections
        bondmap = {}
        for bond in bonds:
            if bond[0] not in bondmap:
                bondmap[bond[0]] = set()
            bondmap[bond[0]].add(bond[1])

        # get lengths
        for atom1 in bondmap:
            for atom2 in bondmap[atom1]:
                if atom1 < atom2:
                    lengths.append((atom1, atom2, -1, -1))

        # get angles
        for atom1 in bondmap:
            for atom2 in bondmap[atom1]:
                for atom3 in bondmap[atom2]:
                    if atom1 >= atom3:
                        continue
                    angles.append((atom1, atom2, atom3, -1))

        # get torsions
        for atom1 in bondmap:
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
        for improper_root in self.improper_roots:
            if improper_root in block_type.atom_to_idx:
                for atom3 in [block_type.atom_to_idx[improper_root]]:
                    comb = list(permutations(bondmap[atom3], 3))
                    for atom1, atom2, atom4 in comb:
                        improper.append((atom1, atom2, atom3, atom4))

        return (
            lengths,
            angles,
            torsions,
            improper,
        )

    def get_raw_params_for_res(self, res: str):
        if res in self.cart_database.residue_params:
            return (
                self.cart_database.residue_params[res].length_parameters
                + self.cart_database.residue_params[res].angle_parameters
                + self.cart_database.residue_params[res].torsion_parameters
                + self.cart_database.residue_params[res].improper_parameters
                + self.cart_database.residue_params[res].hxltorsion_parameters
            )
        return []

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

    def get_params_for_res(self, res: str):
        cached = self._params_for_res_cache.get(res)
        if cached is not None:
            return cached

        params_by_atom_unique_id = {}

        # Fetch the raw params from the DB
        all_params = self.get_raw_params_for_res(res)

        # res "wildcard" matches by atom name in any residue type; a leading '+'
        # marks an atom across a residue connection. The two axes are independent.
        is_wildcard = res == "wildcard"

        for param in all_params:
            # Format the raw param
            atoms, params = self.get_formatted_atoms_and_params(param)

            for i, atom in enumerate(atoms):
                if atom.startswith(CROSS_RES_PREFIX):
                    atoms[i] = self.get_atom_cross_id_name(
                        atom[len(CROSS_RES_PREFIX) :]
                    )
                elif is_wildcard:
                    atoms[i] = self.get_atom_wildcard_id_name(atom)
                else:
                    atoms[i] = self.get_atom_unique_id_name(res, atom)

            key = tuple(atoms)

            params_by_atom_unique_id[key] = params

        self._params_for_res_cache[res] = params_by_atom_unique_id
        return params_by_atom_unique_id

    def setup_block_type(self, block_type: RefinedResidueType):
        super(CartBondedEnergyTerm, self).setup_block_type(block_type)
        if (
            hasattr(block_type, "cartbonded_annotations")
            and self.hash in block_type.cartbonded_annotations
        ):
            return

        # if hasattr(block_type, "cartbonded_subgraphs"):
        #     assert hasattr(block_type, "cartbonded_subgraph_type_counts")
        #     assert hasattr(block_type, "cartbonded_subgraph_type_offsets")
        #     assert hasattr(block_type, "cartbonded_params")
        #     return

        # Get the subgraphs for this block type
        lengths, angles, torsions, improper = self.find_subgraphs(
            block_type.bond_indices, block_type
        )
        cart_subgraphs = numpy.asarray(lengths + angles + torsions + improper)
        cart_subgraph_type_counts = numpy.array(
            [len(lengths), len(angles), len(torsions) + len(improper)]
        )
        cart_subgraph_type_offsets = numpy.array(
            [
                0,
                cart_subgraph_type_counts[0],
                cart_subgraph_type_counts[0] + cart_subgraph_type_counts[1],
            ]
        )

        # Fetch the params from the database, updating the atom id store if necessary
        cartbonded_params = self.get_params_for_res(block_type.base_name)
        cb_block_ann = CartBondedBlockAnnotations(
            cartbonded_subgraphs=cart_subgraphs,
            cartbonded_subgraph_type_counts=cart_subgraph_type_counts,
            cartbonded_subgraph_type_offsets=cart_subgraph_type_offsets,
            cartbonded_params=cartbonded_params,
        )
        if not hasattr(block_type, "cartbonded_annotations"):
            setattr(block_type, "cartbonded_annotations", {})
        block_type.cartbonded_annotations[self.hash] = cb_block_ann

    @staticmethod
    def _padded_param_key(key):
        padded = [-1, -1, -1, -1]
        padded[: len(key)] = key
        return tuple(padded)

    @staticmethod
    def _lookup_param_index(key, param_key_to_index):
        return param_key_to_index.get(CartBondedEnergyTerm._padded_param_key(key), -1)

    def _precompute_subgraph_param_indices(
        self,
        packed_block_types,
        subgraph_offsets,
        total_subgraphs,
        param_key_to_index,
    ):
        """Resolve invariant intra-block parameter searches once during setup."""
        atom_unique_ids = packed_block_types.atom_unique_ids.cpu().numpy()
        atom_wildcard_ids = packed_block_types.atom_wildcard_ids.cpu().numpy()
        param_indices = numpy.full(total_subgraphs, -1, dtype=numpy.int32)

        for block_type_index, block_type in enumerate(
            packed_block_types.active_block_types
        ):
            block_params = block_type.cartbonded_annotations[self.hash]
            block_offset = subgraph_offsets[block_type_index]
            unique_ids = atom_unique_ids[block_type_index]
            wildcard_ids = atom_wildcard_ids[block_type_index]
            subgraphs = numpy.asarray(
                block_params.cartbonded_subgraphs, dtype=numpy.int32
            )
            for local_index, subgraph in enumerate(subgraphs):
                n_atoms = 4
                while n_atoms > 0 and subgraph[n_atoms - 1] == -1:
                    n_atoms -= 1
                if n_atoms == 0:
                    continue
                found = -1
                for atom_ids in (unique_ids, wildcard_ids):
                    found = self._lookup_param_index(
                        [int(atom_ids[subgraph[i]]) for i in range(n_atoms)],
                        param_key_to_index,
                    )
                    if found != -1:
                        break
                    found = self._lookup_param_index(
                        [
                            int(atom_ids[subgraph[n_atoms - 1 - i]])
                            for i in range(n_atoms)
                        ],
                        param_key_to_index,
                    )
                    if found != -1:
                        break
                param_indices[block_offset + local_index] = found
        return param_indices

    def setup_packed_block_types(
        self, packed_block_types: PackedBlockTypes
    ):  # noqa: C901
        super(CartBondedEnergyTerm, self).setup_packed_block_types(packed_block_types)

        if not hasattr(packed_block_types, "cartbonded_is_fragment"):
            packed_block_types.cartbonded_is_fragment = torch.as_tensor(
                numpy.asarray(
                    [
                        block_type.is_ligand_fragment
                        for block_type in packed_block_types.active_block_types
                    ],
                    dtype=numpy.int32,
                ),
                device=self.device,
            )
        if (
            hasattr(packed_block_types, "cartbonded_annotations")
            and self.hash in packed_block_types.cartbonded_annotations
        ):
            return

        # if hasattr(packed_block_types, "cartbonded_subgraphs"):
        #     assert hasattr(packed_block_types, "cartbonded_subgraph_offsets")
        #     assert hasattr(packed_block_types, "cartbonded_max_subgraphs_per_block")
        #     assert hasattr(packed_block_types, "cartbonded_atom_unique_id_index")
        #     assert hasattr(packed_block_types, "cartbonded_params_hash_keys")
        #     assert hasattr(packed_block_types, "cartbonded_params_hash_values")
        #     return

        # Aggregate the subgraphs and collect metadata
        total_subgraphs = sum(
            bt.cartbonded_annotations[self.hash].cartbonded_subgraphs.shape[0]
            for bt in packed_block_types.active_block_types
        )
        subgraphs = numpy.full((total_subgraphs, 4), -1, dtype=numpy.int32)
        subgraph_offsets = []
        subgraph_type_counts = []
        subgraph_type_offsets = []
        offset = 0
        max_subgraphs_per_block = 0
        for block_type in packed_block_types.active_block_types:
            subgraph_offsets.append(offset)
            bt_params = block_type.cartbonded_annotations[self.hash]
            n_subgraphs = bt_params.cartbonded_subgraphs.shape[0]
            subgraph_type_counts.append(bt_params.cartbonded_subgraph_type_counts)
            subgraph_type_offsets.append(bt_params.cartbonded_subgraph_type_offsets)
            subgraphs[offset : offset + n_subgraphs] = bt_params.cartbonded_subgraphs
            offset += n_subgraphs

            max_subgraphs_per_block = max(
                max_subgraphs_per_block, offset - subgraph_offsets[-1]
            )
        subgraph_offsets = numpy.asarray(subgraph_offsets, dtype=numpy.int32)
        subgraph_type_counts = numpy.asarray(subgraph_type_counts, dtype=numpy.int32)
        subgraph_type_offsets = numpy.asarray(subgraph_type_offsets, dtype=numpy.int32)

        # Aggregate the params
        # we will be adding new "wildcard" atom ids, so we will copy the
        # atom_unique_id_index annotation
        cbet_atom_unique_id_index = packed_block_types.atom_unique_id_index.copy()

        # get the params not associated with any specific residue
        wildcard_params = self.get_params_for_res("wildcard").items()

        # we have perhaps created new atom ids for this "wildcard" residue name,
        # so we must expand our set of unique atom ids.
        for key, _ in wildcard_params:
            for at in key:
                if at not in cbet_atom_unique_id_index:
                    cbet_atom_unique_id_index[at] = len(cbet_atom_unique_id_index)

        block_param_sets = {
            bt.base_name: bt.cartbonded_annotations[self.hash].cartbonded_params
            for bt in packed_block_types.active_block_types
        }
        for params in block_param_sets.values():
            for key in params:
                for at in key:
                    if at not in cbet_atom_unique_id_index:
                        cbet_atom_unique_id_index[at] = len(cbet_atom_unique_id_index)

        # Collect each parameter key once. Residue variants with the same base
        # name share parameter dictionaries, and hash lookup uses the first
        # inserted value for duplicate keys.
        padded_key = self._padded_param_key
        unique_params = {}
        for params in block_param_sets.values():
            for key_w_str, value in params.items():
                key = tuple(cbet_atom_unique_id_index[at] for at in key_w_str)
                unique_params.setdefault(padded_key(key), (key, value))

        for key_w_str, value in wildcard_params:
            key = tuple(cbet_atom_unique_id_index[at] for at in key_w_str)
            unique_params.setdefault(padded_key(key), (key, value))

        hash_keys, hash_values = make_hashtable_keys_values(len(unique_params), 2, 5, 7)
        param_key_to_index = {}
        for cur_val, (padded, (key, value)) in enumerate(unique_params.items()):
            add_to_hashtable(hash_keys, hash_values, cur_val, key, value)
            param_key_to_index[padded] = cur_val

        # Intra-block topology and atom naming are fixed for a packed block
        # type. Resolve the exact/reversed/wildcard parameter search once here
        # instead of repeating four hash probes in every scoring invocation.
        subgraph_param_indices = self._precompute_subgraph_param_indices(
            packed_block_types,
            subgraph_offsets,
            total_subgraphs,
            param_key_to_index,
        )

        subgraphs = torch.from_numpy(subgraphs).to(device=self.device)
        subgraph_offsets = torch.from_numpy(subgraph_offsets).to(device=self.device)
        subgraph_type_counts = torch.from_numpy(subgraph_type_counts).to(
            device=self.device
        )
        subgraph_type_offsets = torch.from_numpy(subgraph_type_offsets).to(
            device=self.device
        )
        hash_keys_tensor = torch.from_numpy(hash_keys).to(device=self.device)
        hash_values_tensor = torch.from_numpy(hash_values).to(device=self.device)
        subgraph_param_indices_tensor = torch.from_numpy(subgraph_param_indices).to(
            device=self.device
        )

        cb_pbt_ann = CartBondedPackedBlockTypesAnnotations(
            cartbonded_subgraphs=subgraphs,
            cartbonded_subgraph_offsets=subgraph_offsets,
            cartbonded_subgraph_type_counts=subgraph_type_counts,
            cartbonded_subgraph_type_offsets=subgraph_type_offsets,
            cartbonded_subgraph_param_indices=subgraph_param_indices_tensor,
            cartbonded_max_subgraphs_per_block=max_subgraphs_per_block,
            cartbonded_atom_unique_id_index=cbet_atom_unique_id_index,
            cartbonded_params_hash_keys=hash_keys_tensor,
            cartbonded_params_hash_values=hash_values_tensor,
        )
        if not hasattr(packed_block_types, "cartbonded_annotations"):
            setattr(packed_block_types, "cartbonded_annotations", {})
        packed_block_types.cartbonded_annotations[self.hash] = cb_pbt_ann

    def setup_poses(self, poses: PoseStack):
        super(CartBondedEnergyTerm, self).setup_poses(poses)

    def get_pose_score_term_function(self):
        from tmol.score.cartbonded.potentials import cartbonded_pose_scores

        return cartbonded_pose_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.cartbonded.potentials import cartbonded_rotamer_scores

        return cartbonded_rotamer_scores

    def get_score_term_attributes(self, pose_stack):
        pbt = pose_stack.packed_block_types

        def _t(ts):
            return tuple(map(lambda t: t.to(torch.float), ts))

        pbt_cb_ann = pbt.cartbonded_annotations[self.hash]
        return [
            pose_stack.inter_residue_connections,
            pbt.atom_paths_from_conn,
            pbt.atom_unique_ids,
            pbt.atom_wildcard_ids,
            pbt.cartbonded_is_fragment,
            pbt.atom_cross_ids,
            pbt_cb_ann.cartbonded_params_hash_keys,
            pbt_cb_ann.cartbonded_params_hash_values,
            pbt_cb_ann.cartbonded_subgraphs,
            pbt_cb_ann.cartbonded_subgraph_offsets,
            pbt_cb_ann.cartbonded_subgraph_type_counts,
            pbt_cb_ann.cartbonded_subgraph_type_offsets,
            pbt_cb_ann.cartbonded_subgraph_param_indices,
        ]
