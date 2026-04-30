"""GenBonded energy term: torsional sub-term of cart-bonded using generic
atom-type parameters rather than per-residue atom-name parameters.

Key differences from CartBondedEnergyTerm:
  - Torsional interactions (proper torsions and improper torsions).
  - Parameter lookup is by atom chemical type (e.g. CS, CD, C*, X) rather
    than by (residue_name, atom_name).  The database carries a hierarchy that
    maps each concrete type to a sequence of fall-back types.
  - Intra-block parameters (proper and improper torsions) are resolved at
    setup_block_type time and stored as a single dense array, tagged by type
    (tag=0 proper, tag=1 improper).  Shape per entry: Vec<Int,5> for the
    4-atom subgraph + type tag, Vec<Real,5> for parameters.
  - Inter-block torsion parameters are stored in a hash table keyed by
    (type1, type2, type3, type4, bond_type_int) so that bond-type-specific
    entries are preferred over wildcard ('~') entries.
  - Bond type of the central bond is tracked through the pipeline and used
    for both intra (Python-time lookup) and inter (GPU-time hash lookup).
"""

import torch
import numpy

from typing import List

from tmol.score.atom_type_dependent_term import AtomTypeDependentTerm

from tmol.database import ParameterDatabase

from tmol.score.genbonded.potentials.compiled import (
    genbonded_pose_scores,
    genbonded_rotamer_scores,
)

from tmol.chemical.restypes import BondType, RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

from tmol.score.common.hash_util import (
    make_hashtable_keys_values,
    add_to_hashtable,
)

# Maximum hierarchy depth for any atom type (concrete → class → X).
MAX_HIER_DEPTH = 3

# Bond-type character to integer encoding (must match impl.hh GB_BOND_WILDCARD etc.)
BOND_CHAR_TO_INT = {
    "~": 0,  # wildcard
    "-": 1,  # SINGLE
    "=": 2,  # DOUBLE
    "#": 3,  # TRIPLE
    "@": 4,  # RING
    ":": 5,  # AROMATIC
}

# IntEnum value → single-char representation used in the database file
BOND_TYPE_TO_CHAR = {
    int(BondType.SINGLE): "-",
    int(BondType.DOUBLE): "=",
    int(BondType.TRIPLE): "#",
    int(BondType.RING): "@",
    int(BondType.AROMATIC): ":",
}

GB_WILDCARD_BOND_INT = 0


class GenBondedEnergyTerm(AtomTypeDependentTerm):
    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(GenBondedEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.gen_database = param_db.scoring.genbonded
        self.device = device

        # Build the global type-string → integer-index mapping once.
        # This covers every type string that appears anywhere in the database
        # (concrete, generic, and wildcard), so the mapping is stable and fixed
        # regardless of which block types are later loaded.
        self._type_to_idx = self.gen_database.make_type_to_idx()
        self._all_type_names = self.gen_database.all_type_names()

    @classmethod
    def class_name(cls):
        return "GenBonded"

    @classmethod
    def score_types(cls):
        import tmol.score.terms.genbonded_creator

        return tmol.score.terms.genbonded_creator.GenBondedTermCreator.score_types()

    def n_bodies(self):
        return 2

    def find_torsion_subgraphs(self, bonds):
        """Return list of (i, j, k, l) tuples for all proper torsions in *bonds*.

        Atoms are represented as local indices within the block.
        """
        torsions = []

        # Build adjacency map (bidirectional)
        bondmap = {}
        for b in bonds:
            bondmap.setdefault(b[0], set()).add(b[1])
            bondmap.setdefault(b[1], set()).add(b[0])

        for atom1 in bondmap:
            for atom2 in bondmap[atom1]:
                for atom3 in bondmap[atom2]:
                    if atom3 == atom1:
                        continue
                    for atom4 in bondmap[atom3]:
                        if atom4 == atom2:
                            continue
                        # Canonical ordering: atom1 < atom4 to avoid duplicates
                        if atom1 >= atom4:
                            continue
                        torsions.append((atom1, atom2, atom3, atom4))

        return torsions

    def find_improper_subgraphs(self, bonds):
        """Return list of (center, n1, n2, n3) tuples for all improper torsions.

        An atom is an improper center if it has exactly 3 bonded neighbors.
        The three neighbor indices are returned in sorted (canonical) order.
        """
        # Build adjacency map (bidirectional)
        bondmap = {}
        for b in bonds:
            bondmap.setdefault(b[0], set()).add(b[1])
            bondmap.setdefault(b[1], set()).add(b[0])

        impropers = []
        for center, neighbors in bondmap.items():
            if len(neighbors) == 3:
                n1, n2, n3 = sorted(neighbors)
                impropers.append((center, n1, n2, n3))

        return impropers

    def get_atom_chem_type(self, block_type: RefinedResidueType, atom_idx: int) -> str:
        """Return the chemical atom type string for *atom_idx* in *block_type*."""
        return block_type.atoms[atom_idx].atom_type

    def resolve_torsion_params(self, block_type: RefinedResidueType, torsions):
        """For each torsion tuple (i,j,k,l), look up its genbonded parameters.

        Returns (kept_torsions, params) where:
          kept_torsions – filtered list of (i,j,k,l) tuples that had a DB match
          params        – numpy float32 array of shape (N_kept, 5):
                          columns [k1, k2, k3, k4, offset]

        Torsions with no matching database entry are dropped from the output.
        The central bond (j,k) bond type is looked up from block_type.bond_to_type
        and passed to find_torsion_params for bond-aware matching.
        """
        kept = []
        rows = []

        for i, j, k, l in torsions:
            t1 = self.get_atom_chem_type(block_type, i)
            t2 = self.get_atom_chem_type(block_type, j)
            t3 = self.get_atom_chem_type(block_type, k)
            t4 = self.get_atom_chem_type(block_type, l)

            # Look up the bond type for the central bond (j,k).
            bond_type_int = block_type.bond_to_type.get(
                (int(j), int(k)), int(BondType.SINGLE)
            )
            bond_char = BOND_TYPE_TO_CHAR.get(bond_type_int, "-")

            entry = self.gen_database.find_torsion_params(t1, t2, t3, t4, bond_char)
            if entry is not None:
                kept.append((i, j, k, l))
                rows.append([entry.k1, entry.k2, entry.k3, entry.k4, entry.offset])

        if rows:
            params = numpy.array(rows, dtype=numpy.float32)
        else:
            params = numpy.zeros((0, 5), dtype=numpy.float32)

        return kept, params

    def resolve_improper_params(self, block_type: RefinedResidueType, impropers):
        """For each improper tuple (center, n1, n2, n3), look up parameters.

        Returns (kept_impropers, params) where:
          kept_impropers – filtered list of tuples that had a DB match
          params         – numpy float32 array of shape (N_kept, 2): [k, delta]

        Impropers with no matching database entry are dropped.
        """
        kept = []
        rows = []

        for quad in impropers:
            center, n1, n2, n3 = quad
            tc = self.get_atom_chem_type(block_type, center)
            t1 = self.get_atom_chem_type(block_type, n1)
            t2 = self.get_atom_chem_type(block_type, n2)
            t3 = self.get_atom_chem_type(block_type, n3)

            entry = self.gen_database.find_improper_params(tc, t1, t2, t3)
            if entry is not None:
                kept.append(quad)
                rows.append([entry.k, entry.delta])

        if rows:
            params = numpy.array(rows, dtype=numpy.float32)
        else:
            params = numpy.zeros((0, 2), dtype=numpy.float32)

        return kept, params

    def atom_hierarchy_indices(self, atom_type: str) -> List[int]:
        """Return a list of up to MAX_HIER_DEPTH type indices for *atom_type*.

        The list goes from most specific to most generic.  Padded with -1 to
        reach MAX_HIER_DEPTH elements.
        """
        hierarchy = self.gen_database.hierarchy_for(atom_type)
        result = []
        for ht in hierarchy[:MAX_HIER_DEPTH]:
            idx = self._type_to_idx.get(ht, -1)
            result.append(idx)
        # Pad to MAX_HIER_DEPTH
        while len(result) < MAX_HIER_DEPTH:
            result.append(-1)
        return result

    # ------------------------------------------------------------------
    # Block-type setup
    # ------------------------------------------------------------------

    def setup_block_type(self, block_type: RefinedResidueType):
        super(GenBondedEnergyTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "genbonded_intra_subgraphs"):
            assert hasattr(block_type, "genbonded_intra_params")
            assert hasattr(block_type, "genbonded_atom_type_hierarchy")
            return

        # --- Proper torsions ---
        all_torsions = self.find_torsion_subgraphs(block_type.bond_indices)
        kept_torsions, torsion_params = self.resolve_torsion_params(
            block_type, all_torsions
        )

        # --- Improper torsions ---
        all_impropers = self.find_improper_subgraphs(block_type.bond_indices)
        kept_impropers, improper_params = self.resolve_improper_params(
            block_type, all_impropers
        )

        # --- Combine into single tagged tensor ---
        # Layout: Vec<Int,5>  = [tag, a0, a1, a2, a3]
        #           tag=0 → proper torsion  (atoms: i,j,k,l)
        #           tag=1 → improper torsion (atoms: center, n1, n2, n3)
        # Params: Vec<Real,5>
        #           proper:   [k1, k2, k3, k4, offset]
        #           improper: [k, delta, 0, 0, 0]
        combined_subgraphs = []
        combined_params = []

        for (i, j, k, l), p in zip(kept_torsions, torsion_params):
            combined_subgraphs.append([0, int(i), int(j), int(k), int(l)])
            combined_params.append(
                [float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])]
            )

        for (c, n1, n2, n3), p in zip(kept_impropers, improper_params):
            combined_subgraphs.append([1, int(c), int(n1), int(n2), int(n3)])
            combined_params.append([float(p[0]), float(p[1]), 0.0, 0.0, 0.0])

        if combined_subgraphs:
            intra_subgraphs = numpy.asarray(combined_subgraphs, dtype=numpy.int32)
            intra_params = numpy.asarray(combined_params, dtype=numpy.float32)
        else:
            intra_subgraphs = numpy.zeros((0, 5), dtype=numpy.int32)
            intra_params = numpy.zeros((0, 5), dtype=numpy.float32)

        setattr(block_type, "genbonded_intra_subgraphs", intra_subgraphs)
        setattr(block_type, "genbonded_intra_params", intra_params)

        # --- Per-atom hierarchy index array ---
        # Shape: (n_atoms, MAX_HIER_DEPTH) – integer indices into all_type_names.
        # Allows the GPU kernel to walk the type hierarchy for inter-block
        # torsion lookups.
        n_atoms = len(block_type.atoms)
        hier_arr = numpy.full((n_atoms, MAX_HIER_DEPTH), -1, dtype=numpy.int32)
        for atom_idx in range(n_atoms):
            atype = self.get_atom_chem_type(block_type, atom_idx)
            hier_arr[atom_idx] = self.atom_hierarchy_indices(atype)

        setattr(block_type, "genbonded_atom_type_hierarchy", hier_arr)

    # ------------------------------------------------------------------
    # Packed-block-types setup
    # ------------------------------------------------------------------

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(GenBondedEnergyTerm, self).setup_packed_block_types(packed_block_types)
        if hasattr(packed_block_types, "genbonded_intra_subgraphs"):
            assert hasattr(packed_block_types, "genbonded_intra_subgraph_offsets")
            assert hasattr(packed_block_types, "genbonded_intra_params")
            assert hasattr(packed_block_types, "genbonded_atom_type_hierarchy")
            assert hasattr(packed_block_types, "genbonded_connection_bond_types")
            assert hasattr(packed_block_types, "genbonded_inter_torsion_hash_keys")
            assert hasattr(packed_block_types, "genbonded_inter_torsion_hash_values")
            return

        block_types = packed_block_types.active_block_types
        n_block_types = len(block_types)

        # ------------------------------------------------------------------
        # 1. Aggregate intra-block subgraphs (proper + improper, combined).
        # ------------------------------------------------------------------
        total_intra = sum(bt.genbonded_intra_subgraphs.shape[0] for bt in block_types)
        # TPack does not like stride 0 — ensure at least 1 row.
        total_intra = max(total_intra, 1)

        intra_subgraphs = numpy.full((total_intra, 5), -1, dtype=numpy.int32)
        intra_params = numpy.zeros((total_intra, 5), dtype=numpy.float32)
        intra_offsets = []
        offset = 0
        for bt in block_types:
            intra_offsets.append(offset)
            n = bt.genbonded_intra_subgraphs.shape[0]
            if n > 0:
                intra_subgraphs[offset : offset + n] = bt.genbonded_intra_subgraphs
                intra_params[offset : offset + n] = bt.genbonded_intra_params
            offset += n

        # ------------------------------------------------------------------
        # 2. Per-atom type hierarchy tensor.
        #
        # Shape: (n_block_types, max_atoms, MAX_HIER_DEPTH) int32.
        # Needed by the GPU kernel for inter-block hash-table lookup.
        # ------------------------------------------------------------------
        max_atoms = max(
            (bt.genbonded_atom_type_hierarchy.shape[0] for bt in block_types),
            default=0,
        )
        atom_hier = numpy.full(
            (n_block_types, max(max_atoms, 1), MAX_HIER_DEPTH), -1, dtype=numpy.int32
        )
        for bt_idx, bt in enumerate(block_types):
            h = bt.genbonded_atom_type_hierarchy  # (n_atoms, MAX_HIER_DEPTH)
            n = h.shape[0]
            atom_hier[bt_idx, :n, :] = h

        # ------------------------------------------------------------------
        # 3. Connection bond-type tensor.
        #
        # Shape: (n_block_types, max_n_conns) int32.
        # gen_connection_bond_types[bt][conn_idx] = bond type int (0=wildcard,
        # 1=SINGLE, 2=DOUBLE, etc.) for the bond crossing connection conn_idx.
        # Used by the GPU kernel to encode the central bond in hash lookups.
        # ------------------------------------------------------------------
        max_n_conns = max(
            (len(bt.connections) for bt in block_types),
            default=0,
        )
        conn_bond_types = numpy.zeros(
            (n_block_types, max(max_n_conns, 1)), dtype=numpy.int32
        )
        for bt_idx, bt in enumerate(block_types):
            n_conns = len(bt.connections)
            if n_conns > 0:
                conn_bond_types[bt_idx, :n_conns] = bt.connection_bond_types

        # ------------------------------------------------------------------
        # 4. Inter-block torsion hash table.
        #
        # Keyed by (type_idx_1, type_idx_2, type_idx_3, type_idx_4, bond_type_int).
        # 5 key elements + 1 value-index slot → Vec<Int,6> in the C++ kernel,
        # matching hash_lookup<Int, 5, D>.
        #
        # Wildcard entries (bond='~') are stored under bond_type_int=0.
        # At GPU time the kernel tries the specific bond type first, then
        # falls back to wildcard (GB_BOND_WILDCARD=0).
        #
        # hash_keys:   (n_entries * SCALE, 6) int32
        # hash_values: (n_entries, 5) float32  [k1, k2, k3, k4, offset]
        # ------------------------------------------------------------------
        n_torsion_entries = len(self.gen_database.torsions)
        type_to_idx = self._type_to_idx
        SCALE = 2  # hash table load factor

        if n_torsion_entries > 0:
            # key_len=6: 5 key slots (4 atom types + 1 bond type) + 1 value-index slot.
            # This matches hash_lookup<Int, 5, D> which expects Vec<Int,6>.
            hash_keys, hash_values = make_hashtable_keys_values(
                n_torsion_entries, SCALE, key_len=6, value_len=5
            )
            for val_idx, entry in enumerate(self.gen_database.torsions):
                et1, et2, et3, et4 = entry.atoms
                # Only insert if all 4 types are known.
                if any(t not in type_to_idx for t in (et1, et2, et3, et4)):
                    continue
                bond_int = BOND_CHAR_TO_INT.get(entry.bond, GB_WILDCARD_BOND_INT)
                key = (
                    type_to_idx[et1],
                    type_to_idx[et2],
                    type_to_idx[et3],
                    type_to_idx[et4],
                    bond_int,
                )
                values = (entry.k1, entry.k2, entry.k3, entry.k4, entry.offset)
                add_to_hashtable(hash_keys, hash_values, val_idx, key, values)
        else:
            # Empty hash table placeholders (1 entry, 2-slot table, 6-wide keys).
            hash_keys = numpy.full((2, 6), -1, dtype=numpy.int32)
            hash_values = numpy.zeros((1, 5), dtype=numpy.float32)

        # ------------------------------------------------------------------
        # 5. Store everything on packed_block_types.
        # ------------------------------------------------------------------
        def to_dev(arr):
            return torch.from_numpy(arr).to(device=self.device)

        setattr(
            packed_block_types,
            "genbonded_intra_subgraphs",
            to_dev(intra_subgraphs),
        )
        setattr(
            packed_block_types,
            "genbonded_intra_subgraph_offsets",
            to_dev(numpy.asarray(intra_offsets, dtype=numpy.int32)),
        )
        setattr(
            packed_block_types,
            "genbonded_intra_params",
            to_dev(intra_params),
        )
        setattr(
            packed_block_types,
            "genbonded_atom_type_hierarchy",
            to_dev(atom_hier),
        )
        setattr(
            packed_block_types,
            "genbonded_connection_bond_types",
            to_dev(conn_bond_types),
        )
        setattr(
            packed_block_types,
            "genbonded_inter_torsion_hash_keys",
            to_dev(hash_keys),
        )
        setattr(
            packed_block_types,
            "genbonded_inter_torsion_hash_values",
            to_dev(hash_values),
        )

    # ------------------------------------------------------------------
    # Pose setup
    # ------------------------------------------------------------------

    def setup_poses(self, poses: PoseStack):
        super(GenBondedEnergyTerm, self).setup_poses(poses)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_pose_score_term_function(self):
        return genbonded_pose_scores

    def get_rotamer_score_term_function(self):
        return genbonded_rotamer_scores

    def get_score_term_attributes(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types

        return [
            pose_stack.inter_residue_connections,
            pbt.atom_paths_from_conn,
            pbt.genbonded_intra_subgraphs,
            pbt.genbonded_intra_subgraph_offsets,
            pbt.genbonded_intra_params,
            pbt.genbonded_atom_type_hierarchy,
            pbt.genbonded_connection_bond_types,
            pbt.genbonded_inter_torsion_hash_keys,
            pbt.genbonded_inter_torsion_hash_values,
        ]
