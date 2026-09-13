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
    (type1, type2, type3, type4, bond_bin), with the same parameter
    priority and reversed matches as the CPU database lookup.
  - Bond type of the central bond is tracked through the pipeline and used
    for both intra (Python-time lookup) and inter (GPU-time hash lookup).
"""

import math
import weakref
from dataclasses import dataclass
from itertools import permutations

import torch
import numpy

from typing import List

from tmol.score import AtomTypeDependentTerm

from tmol.database import ParameterDatabase
from tmol.database.scoring._genbonded import _bond_btidx

from tmol.score.genbonded.potentials import (
    genbonded_pose_scores,
    genbonded_rotamer_scores,
)

from tmol.chemical import BondType, RefinedResidueType
from tmol.chemical import MAX_PATHS_FROM_CONNECTION
from tmol.pose import (
    PackedBlockTypes,
    PoseStack,
)
from tmol.score.common import (
    make_hashtable_keys_values,
    add_to_hashtable,
)
from tmol.score.na_torsion import scored_torsion_bonds
from tmol.score.backbone_torsion import omega_connection
from tmol.utility.weak_identity_cache import WeakIdentityLRU

# Maximum hierarchy depth for any atom type (concrete -> class -> X).
MAX_HIER_DEPTH = 4

_INTER_TABLE_CACHE = WeakIdentityLRU()


@dataclass(frozen=True, slots=True)
class _ParameterAnnotation:
    database: weakref.ReferenceType
    elements: tuple
    device: object
    values: tuple


_PACKED_FIELDS = (
    "genbonded_intra_subgraphs",
    "genbonded_intra_subgraph_offsets",
    "genbonded_intra_params",
    "genbonded_atom_type_hierarchy",
    "genbonded_atom_is_rosetta",
    "genbonded_connection_bond_bins",
    "genbonded_conn_scored_elsewhere",
    "genbonded_source_atom_index",
    "genbonded_source_block_type_index",
    "genbonded_inter_torsion_hash_keys",
    "genbonded_inter_torsion_hash_values",
    "genbonded_inter_improper_hash_keys",
    "genbonded_inter_improper_hash_values",
)


class GenBondedEnergyTerm(AtomTypeDependentTerm):
    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(GenBondedEnergyTerm, self).__init__(param_db=param_db, device=device)
        self.gen_database = param_db.scoring.genbonded
        self.device = device

        # Build the global type-string -> integer-index mapping once.
        # This covers every type string that appears anywhere in the database
        # (concrete, generic, and wildcard), so the mapping is stable and fixed
        # regardless of which block types are later loaded.
        self._type_to_idx = self.gen_database.make_type_to_idx()
        self._element_for_atom_type = {
            at.name: at.element for at in param_db.chemical.atom_types
        }
        self._element_key = tuple(sorted(self._element_for_atom_type.items()))
        self._database_ref = weakref.ref(self.gen_database)

    @classmethod
    def class_name(cls):
        return "GenBonded"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._genbonded_creator

        return tmol.score.terms._genbonded_creator.GenBondedTermCreator.score_types()

    def n_bodies(self):
        return 2

    @staticmethod
    def _calculate_offset(k1, k2, k3, k4):
        """Match Rosetta's GenTorsionParams::calculate_offset().

        If more than one of k1,k2,k3 is non-zero, scan energy at 5-degree
        intervals and return the negative of the minimum so the potential
        is zero at its lowest point. Only invoked when the database offset
        is zero (non-zero database offsets override this entirely).
        """
        nk_nonzero = sum(1 for k in (k1, k2, k3) if abs(k) > 1e-6)
        if nk_nonzero <= 1:
            return 0.0
        t = numpy.arange(-180.0, 180.0, 5.0) * (math.pi / 180.0)
        e = (
            k1 * (1 + numpy.cos(t))
            + k2 * (1 + numpy.cos(2 * t))
            + k3 * (1 + numpy.cos(3 * t))
            + k4 * (1 + numpy.cos(4 * t))
        )
        if k1 < 0:
            e += -2.0 * k1
        if k2 < 0:
            e += -2.0 * k2
        if k3 < 0:
            e += -2.0 * k3
        if k4 < 0:
            e += -2.0 * k4
        return -float(e.min())

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
        """Return the generic lookup reference, independently of ownership."""
        atom = block_type.atoms[atom_idx]
        return atom.atom_type if atom.genbonded_type is None else atom.genbonded_type

    def _validate_generic_references(self, block_type):
        for atom in block_type.atoms:
            reference = atom.genbonded_type
            if reference is None:
                continue
            if (
                reference not in self.gen_database.atom_hierarchy
                or reference not in self._element_for_atom_type
                or self._element_for_atom_type[reference]
                != self._element_for_atom_type[atom.atom_type]
            ):
                raise ValueError(
                    f"{block_type.name} atom {atom.name}: invalid genbonded_type "
                    f"{reference!r} for atom type {atom.atom_type!r}"
                )

    def resolve_torsion_params(self, block_type: RefinedResidueType, torsions):
        """For each torsion tuple (i,j,k,l), look up its genbonded parameters.

        Returns (kept_torsions, params) where:
          kept_torsions : filtered list of (i,j,k,l) tuples that had a DB match
          params        : numpy float32 array of shape (N_kept, 5):
                          columns [k1, k2, k3, k4, offset]

        A torsion whose two central atoms are both Rosetta-typed belongs to the
        Rosetta terms (rama, omega, dunbrack, cartbonded) and is skipped here, so
        that every movable torsion is constrained exactly once.  Atom types do
        not settle every case: na_torsion claims its torsions by name, and the
        glycosidic bond of a modified nucleotide is mixed-typed, so the bonds it
        scores are skipped too.

        Torsions with no matching database entry are dropped from the output.
        The central bond (j,k) bond type is looked up from block_type.bond_to_type
        and passed to find_torsion_params for bond-aware matching.
        """
        kept = []
        rows = []
        rosetta_typed = self.gen_database.rosetta_typed
        na_bonds = scored_torsion_bonds(block_type, self._element_for_atom_type)

        for i, j, k, last in torsions:
            t1 = self.get_atom_chem_type(block_type, i)
            t2 = self.get_atom_chem_type(block_type, j)
            t3 = self.get_atom_chem_type(block_type, k)
            t4 = self.get_atom_chem_type(block_type, last)

            if (
                block_type.atoms[j].atom_type in rosetta_typed
                and block_type.atoms[k].atom_type in rosetta_typed
            ):
                continue
            if frozenset((int(j), int(k))) in na_bonds:
                continue

            # Look up bond type and ring membership for the central bond (j,k).
            bond_type_int = block_type.bond_to_type.get(
                (int(j), int(k)), int(BondType.SINGLE)
            )
            is_ring = block_type.bond_to_ringness.get((int(j), int(k)), False)

            # find_torsion_params tries both forward and reversed directions
            # internally and returns the most specific match.
            entry = self.gen_database.find_torsion_params(
                t1, t2, t3, t4, bond_type_int, is_ring
            )
            if entry is not None:
                kept.append((i, j, k, last))
                # Rosetta's calculate_offset() zeros the minimum when the
                # database offset is 0 and nk_nonzero(k1,k2,k3) > 1.
                # Non-zero database offsets override this entirely.
                offset = entry.offset
                if abs(offset) < 1e-6:
                    offset = self._calculate_offset(
                        entry.k1, entry.k2, entry.k3, entry.k4
                    )
                rows.append([entry.k1, entry.k2, entry.k3, entry.k4, offset])

        if rows:
            params = numpy.array(rows, dtype=numpy.float32)
        else:
            params = numpy.zeros((0, 5), dtype=numpy.float32)

        return kept, params

    def resolve_improper_params(self, block_type: RefinedResidueType, impropers):
        """For each improper tuple (center, n1, n2, n3), look up parameters.

        Returns (kept_impropers, params) where:
          kept_impropers : filtered list of tuples that had a DB match
          params         : numpy float32 array of shape (N_kept, 2): [k, delta]

        Impropers with no matching database entry are dropped.

        A Rosetta-typed center belongs to the Rosetta terms even when it has
        an explicit generic lookup reference for neighboring interactions.
        """
        kept = []
        rows = []

        for quad in impropers:
            center, n1, n2, n3 = quad
            if block_type.atoms[center].atom_type in self.gen_database.rosetta_typed:
                continue
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
        previous = getattr(block_type, "_genbonded_parameters", None)
        if (
            previous is not None
            and previous.database() is self.gen_database
            and previous.elements == self._element_key
        ):
            return previous

        # Validate once, before publishing any partially constructed annotation.
        self._validate_generic_references(block_type)

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
        #           tag=0 -> proper torsion  (atoms: i,j,k,l)
        #           tag=1 -> improper torsion (atoms: center, n1, n2, n3)
        # Params: Vec<Real,5>
        #           proper:   [k1, k2, k3, k4, offset]
        #           improper: [k, delta, 0, 0, 0]
        combined_subgraphs = []
        combined_params = []

        for atoms, p in zip(kept_torsions, torsion_params):
            combined_subgraphs.append([0, *map(int, atoms)])
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

        # --- Per-atom hierarchy index array ---
        # Shape: (n_atoms, MAX_HIER_DEPTH) – integer indices into all_type_names.
        # Allows the GPU kernel to walk the type hierarchy for inter-block
        # torsion lookups.
        n_atoms = len(block_type.atoms)
        hier_arr = numpy.full((n_atoms, MAX_HIER_DEPTH), -1, dtype=numpy.int32)
        for atom_idx in range(n_atoms):
            atype = self.get_atom_chem_type(block_type, atom_idx)
            hier_arr[atom_idx] = self.atom_hierarchy_indices(atype)

        annotation = _ParameterAnnotation(
            self._database_ref,
            self._element_key,
            None,
            (intra_subgraphs, intra_params, hier_arr),
        )
        for name, value in zip(
            (
                "genbonded_intra_subgraphs",
                "genbonded_intra_params",
                "genbonded_atom_type_hierarchy",
            ),
            annotation.values,
        ):
            setattr(block_type, name, value)
        block_type._genbonded_parameters = annotation
        return annotation

    # ------------------------------------------------------------------
    # Packed-block-types setup
    # ------------------------------------------------------------------

    def setup_packed_block_types(  # noqa: C901
        self, packed_block_types: PackedBlockTypes
    ):
        super(GenBondedEnergyTerm, self).setup_packed_block_types(packed_block_types)
        previous = getattr(packed_block_types, "_genbonded_parameters", None)
        if (
            previous is not None
            and previous.database() is self.gen_database
            and previous.elements == self._element_key
            and previous.device == self.device
        ):
            return previous

        block_types = packed_block_types.active_block_types
        # Capture each returned annotation, so another setup cannot switch the
        # parameters while this packed set is being assembled.
        block_values = [self.setup_block_type(bt).values for bt in block_types]
        n_block_types = len(block_types)

        # ------------------------------------------------------------------
        # 1. Aggregate intra-block subgraphs (proper + improper, combined).
        # ------------------------------------------------------------------
        total_intra = sum(values[0].shape[0] for values in block_values)
        # TPack does not like stride 0 — ensure at least 1 row.
        total_intra = max(total_intra, 1)

        intra_subgraphs = numpy.full((total_intra, 5), -1, dtype=numpy.int32)
        intra_params = numpy.zeros((total_intra, 5), dtype=numpy.float32)
        intra_offsets = []
        offset = 0
        for graphs, parameters, _hierarchy in block_values:
            intra_offsets.append(offset)
            n = graphs.shape[0]
            if n > 0:
                intra_subgraphs[offset : offset + n] = graphs
                intra_params[offset : offset + n] = parameters
            offset += n

        # ------------------------------------------------------------------
        # 2. Per-atom type hierarchy tensor.
        #
        # Shape: (n_block_types, max_atoms, MAX_HIER_DEPTH) int32.
        # Needed by the GPU kernel for inter-block hash-table lookup.
        # ------------------------------------------------------------------
        max_atoms = max(
            (values[2].shape[0] for values in block_values),
            default=0,
        )
        atom_hier = numpy.full(
            (n_block_types, max(max_atoms, 1), MAX_HIER_DEPTH), -1, dtype=numpy.int32
        )
        for bt_idx, values in enumerate(block_values):
            h = values[2]  # (n_atoms, MAX_HIER_DEPTH)
            n = h.shape[0]
            atom_hier[bt_idx, :n, :] = h

        # Which atoms the Rosetta terms score directly. A torsion whose two
        # central atoms are both of this kind belongs to them, so the kernel
        # applies the same rule the intra-block path does.
        rosetta_typed = self.gen_database.rosetta_typed
        atom_is_rosetta = numpy.zeros(
            (n_block_types, max(max_atoms, 1)), dtype=numpy.int32
        )
        for bt_idx, bt in enumerate(block_types):
            for atom_idx, atom in enumerate(bt.atoms):
                atom_is_rosetta[bt_idx, atom_idx] = atom.atom_type in rosetta_typed

        # Slot zero describes the connection; later slots describe each path's
        # local central bond for 3+1 torsions, including its ring membership.
        max_n_conns = max(
            (len(bt.connections) for bt in block_types),
            default=0,
        )
        conn_bond_bins = numpy.zeros(
            (n_block_types, max(max_n_conns, 1), MAX_PATHS_FROM_CONNECTION + 1),
            dtype=numpy.int32,
        )
        # backbone_torsion claims the torsion across one connection by name, and
        # that bond can be mixed-typed, which atom types alone cannot detect.
        conn_scored_elsewhere = numpy.zeros(
            (n_block_types, max(max_n_conns, 1)), dtype=numpy.int32
        )
        for bt_idx, bt in enumerate(block_types):
            n_conns = len(bt.connections)
            for conn, paths in enumerate(bt.atom_paths_from_conn):
                conn_bond_bins[bt_idx, conn, 0] = _bond_btidx(
                    int(bt.connection_bond_types[conn]), False
                )
                for path, (first, second, third) in enumerate(paths):
                    if third < 0:
                        continue
                    bond = (int(first), int(second))
                    conn_bond_bins[bt_idx, conn, path + 1] = _bond_btidx(
                        bt.bond_to_type[bond], bt.bond_to_ringness[bond]
                    )
            omega_conn = omega_connection(bt)
            if 0 <= omega_conn < n_conns:
                conn_scored_elsewhere[bt_idx, omega_conn] = 1

        source_atom_index = numpy.full(
            (n_block_types, max(max_atoms, 1)), -1, dtype=numpy.int32
        )
        source_by_base = {bt.name: bt for bt in block_types if bt.name == bt.base_name}
        block_type_index_by_name = {
            bt.name: bt_idx for bt_idx, bt in enumerate(block_types)
        }
        source_block_type_index = numpy.empty(n_block_types, dtype=numpy.int32)
        for bt_idx, bt in enumerate(block_types):
            if bt.is_ligand_fragment:
                source = source_by_base.get(bt.base_name, bt)
            else:
                source = bt
            source_block_type_index[bt_idx] = block_type_index_by_name[source.name]
            for atom_idx, atom in enumerate(bt.atoms):
                source_atom_index[bt_idx, atom_idx] = source.atom_to_idx.get(
                    atom.name, atom_idx
                )

        def to_dev(arr):
            return torch.from_numpy(arr).to(device=self.device)

        inter_tables = _INTER_TABLE_CACHE.get_or_create(
            self.gen_database, self.device, self._build_inter_tables
        )
        values = (
            tuple(
                to_dev(arr)
                for arr in (
                    intra_subgraphs,
                    numpy.asarray(intra_offsets, dtype=numpy.int32),
                    intra_params,
                    atom_hier,
                    atom_is_rosetta,
                    conn_bond_bins,
                    conn_scored_elsewhere,
                    source_atom_index,
                    source_block_type_index,
                )
            )
            + inter_tables
        )
        annotation = _ParameterAnnotation(
            self._database_ref, self._element_key, self.device, values
        )
        for name, value in zip(_PACKED_FIELDS, values):
            setattr(packed_block_types, name, value)
        packed_block_types._genbonded_parameters = annotation
        return annotation

    def _build_inter_tables(self):
        """Database-wide tables, shared without retaining their database owner."""
        # The CPU index owns bond coverage, reversed matches, and precedence.
        # Values sorted by that priority let native lookup compare integer ranks.
        index = self.gen_database._torsion_index
        ranked = sorted(
            {
                (priority, entry)
                for bucket in index.values()
                for priority, _, entry in bucket
            },
            key=lambda row: row[0],
        )
        ranks = {}
        for rank, (_, entry) in enumerate(ranked):
            ranks.setdefault(entry, rank)
        keys = {}
        for types, bucket in index.items():
            encoded = tuple(self._type_to_idx[t] for t in types)
            for _, bins, entry in bucket:
                for bond_bin in bins:
                    key = (*encoded, bond_bin)
                    keys.setdefault(key, ranks[entry])
        SCALE = 2
        type_to_idx = self._type_to_idx
        hash_keys = numpy.full((max(len(keys) * SCALE, 2), 6), -1, dtype=numpy.int32)
        hash_values = numpy.zeros((max(len(ranked), 1), 5), dtype=numpy.float32)
        for rank, (_, entry) in enumerate(ranked):
            offset = entry.offset
            if abs(offset) < 1e-6:
                offset = self._calculate_offset(entry.k1, entry.k2, entry.k3, entry.k4)
            hash_values[rank] = (entry.k1, entry.k2, entry.k3, entry.k4, offset)
        for key, rank in keys.items():
            add_to_hashtable(hash_keys, hash_values, rank, key, hash_values[rank])

        # Inter-block impropers are keyed by center type followed by an ordered
        # permutation of the three neighbor types. Store all six permutations;
        # the kernel can then use the same hierarchy walk as proper torsions
        # without special-casing unordered neighbors.
        n_improper_entries = len(self.gen_database.impropers) * 6
        if n_improper_entries:
            improper_hash_keys, improper_hash_values = make_hashtable_keys_values(
                n_improper_entries, SCALE, key_len=5, value_len=5
            )
            improper_val_idx = 0
            for entry in self.gen_database.impropers:
                center, n1, n2, n3 = entry.atoms
                if any(
                    atom_type not in type_to_idx for atom_type in (center, n1, n2, n3)
                ):
                    continue
                for neighbors in permutations((n1, n2, n3)):
                    key = (
                        type_to_idx[center],
                        *(type_to_idx[neighbor] for neighbor in neighbors),
                    )
                    values = (entry.k, entry.delta, 0.0, 0.0, 0.0)
                    add_to_hashtable(
                        improper_hash_keys,
                        improper_hash_values,
                        improper_val_idx,
                        key,
                        values,
                    )
                    improper_val_idx += 1
        else:
            improper_hash_keys = numpy.full((2, 5), -1, dtype=numpy.int32)
            improper_hash_values = numpy.zeros((1, 5), dtype=numpy.float32)

        return tuple(
            torch.from_numpy(arr).to(device=self.device)
            for arr in (
                hash_keys,
                hash_values,
                improper_hash_keys,
                improper_hash_values,
            )
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
        parameters = self.setup_packed_block_types(pbt)
        return [
            pose_stack.inter_residue_connections,
            pbt.atom_paths_from_conn,
            *parameters.values,
        ]
