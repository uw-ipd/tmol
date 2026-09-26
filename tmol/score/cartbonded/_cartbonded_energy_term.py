import torch
import numpy
import attrs

from itertools import permutations, product

from tmol.score import AtomTypeDependentTerm
from tmol.score._annotation_cache import (
    AnnotationKey,
    cached_annotation,
    store_annotation,
)

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
    atom_unique_ids: torch.Tensor
    atom_wildcard_ids: torch.Tensor
    atom_cross_ids: torch.Tensor
    cartbonded_params_hash_keys: torch.Tensor
    cartbonded_params_hash_values: torch.Tensor
    connection_hash_keys: torch.Tensor
    connection_spans: torch.Tensor
    connection_paths: torch.Tensor
    atom_is_rosetta: torch.Tensor
    rosetta_typed: frozenset


class CartBondedEnergyTerm(AtomTypeDependentTerm):
    device: torch.device  # = attr.ib()
    improper_roots: set()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(CartBondedEnergyTerm, self).__init__(param_db=param_db, device=device)

        # Improper centres, kept keyed by the residue they were parameterised
        # for. Flattened to bare names they match any block type that happens to
        # share an atom name -- a calcium ion's atom is CA, as an alpha carbon
        # is -- which would hand a ligand improper torsions built from protein
        # parameters.
        def find_improper_roots(db):
            return {
                res: frozenset(
                    imp.atm3.lstrip(CROSS_RES_PREFIX)
                    for imp in params.improper_parameters
                )
                for res, params in db.residue_params.items()
            }

        self.improper_roots = find_improper_roots(param_db.scoring.cartbonded)

        self.cart_database = param_db.scoring.cartbonded
        self.rosetta_typed = frozenset(param_db.scoring.genbonded.rosetta_typed)
        self.hash = self.cart_database.hash
        self._params_for_res_cache = {}
        self._block_annotation_key = AnnotationKey.from_sources(self.cart_database)
        self._packed_annotation_key = AnnotationKey.from_sources(
            param_db.chemical,
            self.cart_database,
            param_db.scoring.genbonded,
            settings=(self.device,),
        )
        self._fragment_annotation_key = AnnotationKey.from_sources(
            settings=(self.device,)
        )
        self._ownership_annotation_key = AnnotationKey.from_sources(
            param_db.scoring.genbonded, settings=(self.device,)
        )

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

        # get improper torsions: those parameterised for this residue, plus the
        # wildcard set that applies to every type. A wildcard root can still
        # land on an atom with too few bonds to centre anything, so the degree
        # is checked rather than assumed.
        roots = self.improper_roots.get(
            self._parameter_name(block_type), frozenset()
        ) | self.improper_roots.get("wildcard", frozenset())
        for improper_root in roots:
            if improper_root in block_type.atom_to_idx:
                atom3 = block_type.atom_to_idx[improper_root]
                neighbors = bondmap.get(atom3, ())
                if len(neighbors) < 3:
                    continue
                for atom1, atom2, atom4 in permutations(neighbors, 3):
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
        cached = cached_annotation(
            block_type, "_cartbonded_annotation", self._block_annotation_key
        )
        if cached is not None:
            return cached

        # Get the subgraphs for this block type
        lengths, angles, torsions, improper = self.find_subgraphs(
            block_type.bond_indices, block_type
        )
        cart_subgraphs = numpy.asarray(
            lengths + angles + torsions + improper, dtype=numpy.int32
        ).reshape(-1, 4)
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
        # An exact patched name replaces this residue's complete CartRes.
        # Keep its atom namespace separate so the unpatched type, other
        # variants and other score terms can retain their existing parameters.
        parameter_name = self._parameter_name(block_type)
        cartbonded_params = self.get_params_for_res(parameter_name)
        cb_block_ann = CartBondedBlockAnnotations(
            cartbonded_subgraphs=cart_subgraphs,
            cartbonded_subgraph_type_counts=cart_subgraph_type_counts,
            cartbonded_subgraph_type_offsets=cart_subgraph_type_offsets,
            cartbonded_params=cartbonded_params,
        )
        return store_annotation(
            block_type,
            "_cartbonded_annotation",
            self._block_annotation_key,
            cb_block_ann,
        )

    def _parameter_name(self, block_type):
        if block_type.is_ligand_fragment:
            records = self.cart_database.residue_params
            # Fragment preparations carry the source ligand's complete
            # CartRes. Equal records must share its namespace so paths across
            # a cut still resolve to the original ligand parameters. This
            # also covers separately deserialized copies of those records.
            if records.get(block_type.name) == records.get(block_type.base_name):
                return block_type.base_name
        return (
            block_type.name
            if block_type.name in self.cart_database.residue_params
            else block_type.base_name
        )

    def _cartbonded_atom_ids(self, packed_block_types, id_index):
        """Reuse ordinary IDs unless an exact variant needs its own namespace."""
        ids = packed_block_types.atom_unique_ids
        overrides = [
            (i, bt)
            for i, bt in enumerate(packed_block_types.active_block_types)
            if self._parameter_name(bt) != bt.base_name
        ]
        if not overrides:
            return ids
        array = ids.cpu().numpy().copy()
        for i, bt in overrides:
            for j, atom in enumerate(bt.atoms):
                name = self.get_atom_unique_id_name(bt.name, atom.name)
                array[i, j] = id_index.setdefault(name, len(id_index))
        return torch.from_numpy(array).to(self.device)

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
        atom_unique_ids,
        atom_wildcard_ids,
        block_annotations,
    ):
        """Resolve invariant intra-block parameter searches once during setup."""
        atom_unique_ids = atom_unique_ids.cpu().numpy()
        # Scoped reference IDs, not the block types' own: param_key_to_index is
        # keyed on the scoped IDs and the kernel is handed the same table.
        atom_wildcard_ids = atom_wildcard_ids.cpu().numpy()
        param_indices = numpy.full(total_subgraphs, -1, dtype=numpy.int32)

        for block_type_index, block_params in enumerate(block_annotations):
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

        fragment = cached_annotation(
            packed_block_types,
            "_cartbonded_fragment_annotation",
            self._fragment_annotation_key,
        )
        if fragment is None:
            fragment = torch.as_tensor(
                numpy.asarray(
                    [
                        block_type.is_ligand_fragment
                        for block_type in packed_block_types.active_block_types
                    ],
                    dtype=numpy.int32,
                ),
                device=self.device,
            )
            packed_block_types.cartbonded_is_fragment = fragment
            store_annotation(
                packed_block_types,
                "_cartbonded_fragment_annotation",
                self._fragment_annotation_key,
                fragment,
                fields=("cartbonded_is_fragment",),
            )
        previous = cached_annotation(
            packed_block_types,
            "_cartbonded_annotation",
            self._packed_annotation_key,
        )
        if previous is not None:
            return previous
        # Capture returned block annotations; cache eviction or another setup
        # must not switch the parameters while this packed set is assembled.
        block_annotations = [
            self.setup_block_type(bt) for bt in packed_block_types.active_block_types
        ]

        # Aggregate the subgraphs and collect metadata
        total_subgraphs = sum(
            annotation.cartbonded_subgraphs.shape[0] for annotation in block_annotations
        )
        subgraphs = numpy.full((total_subgraphs, 4), -1, dtype=numpy.int32)
        subgraph_offsets = []
        subgraph_type_counts = []
        subgraph_type_offsets = []
        offset = 0
        max_subgraphs_per_block = 0
        for bt_params in block_annotations:
            subgraph_offsets.append(offset)
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
        atom_unique_ids = self._cartbonded_atom_ids(
            packed_block_types, cbet_atom_unique_id_index
        )

        # get the params not associated with any specific residue
        wildcard_params = self.get_params_for_res("wildcard").items()

        # we have perhaps created new atom ids for this "wildcard" residue name,
        # so we must expand our set of unique atom ids.
        for key, _ in wildcard_params:
            for at in key:
                if at not in cbet_atom_unique_id_index:
                    cbet_atom_unique_id_index[at] = len(cbet_atom_unique_id_index)

        for bt_params in block_annotations:
            for key in bt_params.cartbonded_params:
                for at in key:
                    if at not in cbet_atom_unique_id_index:
                        cbet_atom_unique_id_index[at] = len(cbet_atom_unique_id_index)

        # Collect each parameter key once. Residue variants with the same base
        # name share parameter dictionaries, and hash lookup uses the first
        # inserted value for duplicate keys.
        named_params = {}
        for annotation in block_annotations:
            for key_w_str, value in annotation.cartbonded_params.items():
                named_params.setdefault(key_w_str, value)

        for key_w_str, value in wildcard_params:
            named_params.setdefault(key_w_str, value)
        atom_wildcard_ids, atom_cross_ids = self._reference_params(
            packed_block_types, cbet_atom_unique_id_index, named_params
        )
        padded_key = self._padded_param_key
        hash_keys, hash_values = make_hashtable_keys_values(
            max(len(named_params), 1), 2, 5, 7
        )
        param_key_to_index = {}
        for cur_val, (key_w_str, value) in enumerate(named_params.items()):
            key = tuple(cbet_atom_unique_id_index[at] for at in key_w_str)
            add_to_hashtable(hash_keys, hash_values, cur_val, key, value)
            param_key_to_index[padded_key(key)] = cur_val

        # Intra-block topology and atom naming are fixed for a packed block
        # type. Resolve the exact/reversed/wildcard parameter search once here
        # instead of repeating four hash probes in every scoring invocation.
        subgraph_param_indices = self._precompute_subgraph_param_indices(
            packed_block_types,
            subgraph_offsets,
            total_subgraphs,
            param_key_to_index,
            atom_unique_ids,
            atom_wildcard_ids,
            block_annotations,
        )

        subgraphs = torch.from_numpy(subgraphs).to(device=self.device)
        subgraph_offsets = torch.from_numpy(subgraph_offsets).to(device=self.device)
        subgraph_type_counts = torch.from_numpy(subgraph_type_counts).to(
            device=self.device
        )
        subgraph_type_offsets = torch.from_numpy(subgraph_type_offsets).to(
            device=self.device
        )
        from ._connection_parameters import compile_connection_parameters

        connection_keys, connection_spans, connection_paths, connection_values = (
            compile_connection_parameters(
                self.cart_database.connection_params,
                packed_block_types,
                len(hash_values),
            )
        )
        hash_values = (
            numpy.concatenate((hash_values, connection_values), axis=0)
            if len(connection_values)
            else hash_values
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
            atom_unique_ids=atom_unique_ids,
            atom_wildcard_ids=atom_wildcard_ids,
            atom_cross_ids=atom_cross_ids,
            cartbonded_params_hash_keys=hash_keys_tensor,
            cartbonded_params_hash_values=hash_values_tensor,
            connection_hash_keys=torch.from_numpy(connection_keys).to(self.device),
            connection_spans=torch.from_numpy(connection_spans).to(self.device),
            connection_paths=torch.from_numpy(connection_paths).to(self.device),
            atom_is_rosetta=self._ownership_mask(packed_block_types),
            rosetta_typed=self.rosetta_typed,
        )
        packed_block_types.cartbonded_atom_is_rosetta = cb_pbt_ann.atom_is_rosetta
        return store_annotation(
            packed_block_types,
            "_cartbonded_annotation",
            self._packed_annotation_key,
            cb_pbt_ann,
            fields=("cartbonded_atom_is_rosetta",),
        )

    def _reference_params(self, pbt, atom_ids, params):
        """Compile name fallbacks once, scoped to the actual block type.

        Keep native lookup unchanged: renamed atoms use private IDs with
        exact-name rows first and borrowed chemical-role rows second.
        Unrelated blocks never acquire that alias; ordinary types reuse their
        existing IDs and allocate no additional table.
        """
        changed = [
            (i, bt)
            for i, bt in enumerate(pbt.active_block_types)
            if any(atom.cartbonded_reference for atom in bt.atoms)
        ]
        originals = (pbt.atom_wildcard_ids, pbt.atom_cross_ids)
        if not changed:
            return originals
        tables = []
        for prefix, original in zip(("WILDCARD_ID:", "CROSS_ID:"), originals):
            rows = [
                (key, value)
                for key, value in params.items()
                if any(atom.startswith(prefix) for atom in key)
            ]
            ids = original.cpu().numpy().copy()
            for i, bt in changed:
                exact, borrowed = {}, {}
                for j, atom in enumerate(bt.atoms):
                    name = prefix + atom.name
                    if atom.cartbonded_reference:
                        scoped = prefix + f"{bt.name}:{atom.name}"
                        ids[i, j] = atom_ids.setdefault(scoped, len(atom_ids))
                    else:
                        scoped = name
                    exact[name] = (scoped,)
                    reference = prefix + (atom.cartbonded_reference or atom.name)
                    borrowed.setdefault(reference, []).append(scoped)
                # Exact supplied rows win regardless of parameter ordering.
                for names in (exact, borrowed):
                    for key, value in rows:
                        choices = [
                            names.get(atom, ()) if atom.startswith(prefix) else (atom,)
                            for atom in key
                        ]
                        for mapped in product(*choices):
                            params.setdefault(mapped, value)
            tables.append(torch.as_tensor(ids, device=self.device))
        return tuple(tables)

    def _ownership_mask(self, packed_block_types):
        cached = cached_annotation(
            packed_block_types,
            "_cartbonded_ownership_annotation",
            self._ownership_annotation_key,
        )
        if cached is not None:
            return cached
        # A planarity centre the Rosetta terms type is theirs; one they do
        # not is the generic term's, which carries its own improper for it.
        rosetta_typed = self.rosetta_typed
        bts = packed_block_types.active_block_types
        max_atoms = max((len(bt.atoms) for bt in bts), default=1)
        mask = numpy.zeros((len(bts), max(max_atoms, 1)), dtype=numpy.int32)
        for i, bt in enumerate(bts):
            for j, atom in enumerate(bt.atoms):
                mask[i, j] = atom.atom_type in rosetta_typed
        value = torch.tensor(mask, dtype=torch.int32, device=self.device)
        return store_annotation(
            packed_block_types,
            "_cartbonded_ownership_annotation",
            self._ownership_annotation_key,
            value,
        )

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

        pbt_cb_ann = self.setup_packed_block_types(pbt)
        return [
            pose_stack.inter_residue_connections,
            pbt.atom_paths_from_conn,
            pbt_cb_ann.atom_unique_ids,
            pbt_cb_ann.atom_wildcard_ids,
            pbt_cb_ann.atom_is_rosetta,
            pbt.cartbonded_is_fragment,
            pbt_cb_ann.atom_cross_ids,
            pbt_cb_ann.connection_hash_keys,
            pbt_cb_ann.connection_spans,
            pbt_cb_ann.connection_paths,
            pbt_cb_ann.cartbonded_params_hash_keys,
            pbt_cb_ann.cartbonded_params_hash_values,
            pbt_cb_ann.cartbonded_subgraphs,
            pbt_cb_ann.cartbonded_subgraph_offsets,
            pbt_cb_ann.cartbonded_subgraph_type_counts,
            pbt_cb_ann.cartbonded_subgraph_type_offsets,
            pbt_cb_ann.cartbonded_subgraph_param_indices,
        ]
