"""Refined residue types for tmol_jax.

Ported from ``tmol.chemical.restypes``. ``RefinedResidueType`` is the
workhorse residue definition used throughout scoring, packing, and
kinematics -- it extends the raw ``RawResidueType`` (from
``tmol_jax.database.chemical``) with a large collection of derived
lookup tables and index arrays that the rest of the codebase wants
in constant-time form.

The upstream class is an ``attr.s`` dataclass whose fields have
``default`` factories that compute themselves on-demand from the raw
residue data. We keep that same pattern here: each ``attr.ib(...)``
field has a ``@field.default`` method so the class can be instantiated
with just the raw fields and have the rest computed automatically.

No torch, no numba. Only numpy + scipy.sparse.csgraph for shortest-path.
"""

from __future__ import annotations

import typing
from collections import defaultdict
from typing import (
    Dict,
    List,
    Mapping,
    NewType,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import attr
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph as csgraph

from tmol_jax.database.chemical import RawResidueType

from tmol_jax.chemical.all_bonds import bonds_and_bond_ranges
from tmol_jax.chemical.constants import (
    MAX_PATHS_FROM_CONNECTION,
    MAX_SIG_BOND_SEPARATION,
)
from tmol_jax.chemical.ideal_coords import build_coords_from_icoors

AtomIndex = NewType("AtomIndex", int)
ConnectionIndex = NewType("ConnectionIndex", int)
BondCount = NewType("BondCount", int)

ResName3 = NewType("ResName3", str)
IcoorIndex = NewType("IcoorIndex", int)

# Matches the upstream struct dtype for an "unresolved atom id".
UnresolvedAtomID = Tuple[AtomIndex, ConnectionIndex, BondCount]
uaid_t = np.dtype(
    [
        ("atom_id", np.int32),
        ("conn_id", np.int32),
        ("n_bonds_from_conn", np.int32),
    ]
)


# --------------------------------------------------------------------- #
# RefinedResidueType                                                     #
# --------------------------------------------------------------------- #


@attr.s(auto_attribs=False)
class RefinedResidueType(RawResidueType):
    """Residue type with computed index tables.

    All ``attr.ib`` fields have a default factory, so you can construct
    one with only the raw fields from a ``RawResidueType`` -- the rest
    are computed lazily-at-init from ``self.atoms``, ``self.bonds``,
    ``self.connections``, ``self.icoors``, etc.
    """

    # ---- basic atom index tables --------------------------------------

    atom_names_set: Set[str] = attr.ib()

    @atom_names_set.default
    def _atom_names_set(self) -> Set[str]:
        return {a.name for a in self.atoms}

    atom_to_idx: Mapping[str, AtomIndex] = attr.ib()

    @atom_to_idx.default
    def _setup_atom_to_idx(self) -> Mapping[str, AtomIndex]:
        # plain dict is fine; we never mutate
        return {a.name: i for i, a in enumerate(self.atoms)}

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    aliases_map: Mapping[str, str] = attr.ib()

    @aliases_map.default
    def _setup_atom_aliases_mapping(self) -> Mapping[str, str]:
        # Guard against aliases that shadow real atoms / other aliases.
        aliases: Set[str] = set()
        for a in self.atom_aliases:
            assert (
                a.alt_name not in self.atom_to_idx
            ), f"Alias {a.alt_name!r} shadows atom {a.alt_name!r} on {self.name!r}"
            assert (
                a.alt_name not in aliases
            ), f"Alias {a.alt_name!r} declared twice on {self.name!r}"
            aliases.add(a.alt_name)
        return {a.alt_name: a.name for a in self.atom_aliases}

    coord_dtype: np.dtype = attr.ib()

    @coord_dtype.default
    def _setup_coord_dtype(self) -> np.dtype:
        return np.dtype([(a.name, float, 3) for a in self.atoms])

    # ---- bond indexing -------------------------------------------------

    bond_indices: np.ndarray = attr.ib()

    @bond_indices.default
    def _setup_bond_indices(self) -> np.ndarray:
        """Directed bond edges: each undirected bond (a, b) appears twice,
        as (a, b) and (b, a)."""
        seen: Set[Tuple[int, int]] = set()
        for bond in self.bonds:
            a_name, b_name = bond[:2]
            ai = self.atom_to_idx[a_name]
            bi = self.atom_to_idx[b_name]
            seen.add((ai, bi))
            seen.add((bi, ai))
        bond_array = np.array(sorted(seen), dtype=np.int32).reshape(-1, 2)
        bond_array.flags.writeable = False
        return bond_array

    # ---- connections ---------------------------------------------------

    @property
    def n_conn(self) -> int:
        return len(self.connections)

    connection_to_idx: Mapping[str, AtomIndex] = attr.ib()

    @connection_to_idx.default
    def _setup_connection_to_idx(self) -> Mapping[str, AtomIndex]:
        return {c.name: self.atom_to_idx[c.atom] for c in self.connections}

    connection_to_cidx: Mapping[Optional[str], ConnectionIndex] = attr.ib()

    @connection_to_cidx.default
    def _setup_connection_to_cidx(self) -> Mapping[Optional[str], ConnectionIndex]:
        centries: List[Tuple[Optional[str], int]] = [(None, -1)]
        centries.extend((c.name, i) for i, c in enumerate(self.connections))
        return dict(centries)

    ordered_connection_atoms: np.ndarray = attr.ib()

    @ordered_connection_atoms.default
    def _setup_ordered_connections(self) -> np.ndarray:
        return np.array(
            [self.atom_to_idx[c.atom] for c in self.connections], dtype=np.int32
        )

    # ---- "all bonds" (intra + inter) ----------------------------------
    #
    # Each row is (atom_ind1, atom_ind2_or_-1, conn_id_or_-1).
    # If the partner atom is on this residue, atom_ind2 >= 0 and
    # conn_id == -1. Otherwise atom_ind2 == -1 and conn_id >= 0 is the
    # index into self.connections that points to the partner.

    all_bonds: np.ndarray = attr.ib()

    @all_bonds.default
    def _setup_all_bonds(self) -> np.ndarray:
        bonds, _ = bonds_and_bond_ranges(
            self.n_atoms, self.bond_indices, self.ordered_connection_atoms
        )
        return bonds

    atom_all_bond_ranges: np.ndarray = attr.ib()

    @atom_all_bond_ranges.default
    def _setup_atom_all_bond_ranges(self) -> np.ndarray:
        _, bond_ranges = bonds_and_bond_ranges(
            self.n_atoms, self.bond_indices, self.ordered_connection_atoms
        )
        return bond_ranges

    # ---- polymer connection bookkeeping -------------------------------

    down_connection_ind: int = attr.ib()

    @down_connection_ind.default
    def _setup_down_connection_ind(self) -> int:
        return self.connection_to_cidx.get("down", -1)

    up_connection_ind: int = attr.ib()

    @up_connection_ind.default
    def _setup_up_connection_ind(self) -> int:
        return self.connection_to_cidx.get("up", -1)

    # ---- torsions ------------------------------------------------------

    torsion_to_uaids: Mapping[str, Tuple[UnresolvedAtomID, ...]] = attr.ib()

    @torsion_to_uaids.default
    def _setup_torsion_to_uaids(self) -> Mapping[str, Tuple[UnresolvedAtomID, ...]]:
        torsion_to_uaids: Dict[str, Tuple[UnresolvedAtomID, ...]] = {}
        for tor in self.torsions:
            ats: List[Tuple[int, int, int]] = []
            for at in (tor.a, tor.b, tor.c, tor.d):
                if at.atom is not None:
                    ats.append((self.atom_to_idx[at.atom], -1, -1))
                else:
                    ats.append(
                        (
                            -1,
                            next(
                                i
                                for i, conn in enumerate(self.connections)
                                if conn.name == at.connection
                            ),
                            at.bond_sep_from_conn,
                        )
                    )
            torsion_to_uaids[tor.name] = tuple(ats)
        return torsion_to_uaids

    ordered_torsions: np.ndarray = attr.ib()

    @ordered_torsions.default
    def _setup_ordered_torsions(self) -> np.ndarray:
        ordered_torsions = np.full((len(self.torsions), 4, 3), -1, dtype=np.int32)
        for i, tor in enumerate(self.torsions):
            for j in range(4):
                ordered_torsions[i, j] = np.array(
                    self.torsion_to_uaids[tor.name][j], dtype=np.int32
                )
        return ordered_torsions

    @property
    def n_torsions(self) -> int:
        return int(self.ordered_torsions.shape[0])

    is_torsion_mc: np.ndarray = attr.ib()

    @is_torsion_mc.default
    def _setup_is_torsion_mc(self) -> np.ndarray:
        # A torsion is "main chain" iff every atom in it is a main-chain
        # atom, or is on another residue (represented by atom_ind == -1).
        def all_torsion_atoms_are_mainchain(tor_ind: int) -> bool:
            if not self.properties.polymer.is_polymer:
                return False
            for i in range(4):
                at_i = self.ordered_torsions[tor_ind, i, 0]
                if at_i == -1:
                    continue
                atname = self.atoms[at_i].name
                if atname not in self.properties.polymer.mainchain_atoms:
                    return False
            return True

        return np.array(
            [all_torsion_atoms_are_mainchain(tor) for tor in range(self.n_torsions)],
            dtype=bool,
        )

    mc_torsions: np.ndarray = attr.ib()

    @mc_torsions.default
    def _setup_mc_torsions(self) -> np.ndarray:
        return np.nonzero(self.is_torsion_mc)[0].astype(np.int32)

    @property
    def n_mc_torsions(self) -> int:
        return int(len(self.mc_torsions))

    sc_torsions: np.ndarray = attr.ib()

    @sc_torsions.default
    def _setup_sc_torsions(self) -> np.ndarray:
        return np.nonzero(np.logical_not(self.is_torsion_mc))[0].astype(np.int32)

    @property
    def n_sc_torsions(self) -> int:
        return int(len(self.sc_torsions))

    which_mcsc_torsion: np.ndarray = attr.ib()

    @which_mcsc_torsion.default
    def _setup_which_mcsc_torsion(self) -> np.ndarray:
        w = np.full(self.n_torsions, -1, dtype=np.int32)
        for i in range(self.n_mc_torsions):
            w[self.mc_torsions[i]] = i
        for i in range(self.n_sc_torsions):
            w[self.sc_torsions[i]] = i
        return w

    # ---- path distance matrix -----------------------------------------

    path_distance: np.ndarray = attr.ib()

    @path_distance.default
    def _setup_path_distance(self) -> np.ndarray:
        if self.n_atoms == 0:
            return np.zeros((0, 0), dtype=np.int32)
        bonds_sparse = scipy.sparse.coo_matrix(
            (
                np.ones(self.bond_indices.shape[0], dtype=bool),
                (self.bond_indices[:, 0], self.bond_indices[:, 1]),
            ),
            shape=(self.n_atoms, self.n_atoms),
        )
        pd = csgraph.dijkstra(
            bonds_sparse,
            directed=False,
            unweighted=True,
            limit=MAX_SIG_BOND_SEPARATION,
        )
        pd[pd == np.inf] = MAX_SIG_BOND_SEPARATION
        return pd.astype(np.int32)

    # Alias requested by the task spec -- same object as path_distance.
    @property
    def intrares_path_distance(self) -> np.ndarray:
        return self.path_distance

    # ---- paths from each connection -----------------------------------

    atom_paths_from_conn: np.ndarray = attr.ib()

    @atom_paths_from_conn.default
    def _setup_atom_paths_from_conn(self):  # noqa: C901
        n_conns = len(self.connections)

        atom_paths = np.full(
            (n_conns, MAX_PATHS_FROM_CONNECTION, 3), -1, dtype=np.int32
        )

        if n_conns == 0:
            return atom_paths

        # Create a numpy array with the paths coming from a connection.
        # The first entry will be the immediate atom, followed by the
        # 3 paths coming from that atom, followed by the 3 coming out
        # of each of those in turn. If a path doesn't exist, it is
        # filled with -1s to ensure deterministic indexing of the paths.
        # bonds to follow, built once for every connection; virtual atoms have no
        #    bonded geometry to reach
        virtual = {self.atom_to_idx[name] for name in self.properties.virtual}
        bondmap = defaultdict(list)
        for bond in self.bond_indices:
            if bond[0] not in virtual and bond[1] not in virtual:
                bondmap[bond[0]].append(bond[1])

        def get_paths_length_3(connection):
            paths = np.full((MAX_PATHS_FROM_CONNECTION, 3), -1, dtype=np.int32)
            atom0 = self.atom_to_idx[connection.atom]
            # Add the immediate atom
            paths[0] = (atom0, -1, -1)

            # an atom with more than three partners besides the way back (a
            #    cluster metal or a bridging oxide) keeps the first three
            def partners(atom, back=None):
                out = bondmap[atom]
                rest = [a for a in out if a != back]
                if len(rest) > 3:
                    out = ([back] if back in out else []) + rest[:3]
                return out + [-1] * (3 - len(out))

            idx = 1
            # Add the 3 paths connecting to the immediate atom
            for atom1 in partners(atom0):
                if atom1 != -1:
                    paths[idx] = (atom0, atom1, -1)
                idx += 1

            # Add the 9 paths connecting to the 3 from the previous step
            for atom1 in partners(atom0):
                for atom2 in partners(atom1, atom0):
                    if atom2 != atom0 and atom2 != -1:
                        paths[idx] = (atom0, atom1, atom2)
                    if atom2 != atom0:
                        idx += 1

            return paths

        # construct a list of paths starting from each connection point of length 3 and record the atom indices of the atoms in those paths
        for i, connection in enumerate(self.connections):
            paths = get_paths_length_3(connection)
            atom_paths[i][: len(paths)] = paths

        return atom_paths

    # ---- atom-downstream-of-conn --------------------------------------

    atom_downstream_of_conn: np.ndarray = attr.ib()

    @atom_downstream_of_conn.default
    def _setup_atom_downstream_of_conn(self):
        n_conns = len(self.connections)
        atom_downstream_of_conn = np.full((n_conns, self.n_atoms), -1, dtype=np.int32)
        for i in range(n_conns):
            i_conn_atom = self.atom_to_idx[self.connections[i].atom]
            if self.connections[i].name == "down":
                # walk up through the mainchain atoms
                mc_ats = self.properties.polymer.mainchain_atoms
                if mc_ats is None:
                    # weird case? The user has a "down" connection but no
                    # atoms are part of the mainchain?
                    atom_downstream_of_conn[i, :] = i_conn_atom
                else:
                    assert mc_ats[0] == self.connections[i].atom
                    if len(mc_ats) == 1:
                        # A cap has no atom farther along its backbone. Keep
                        # the undefined-depth sentinel instead of repeating
                        # the connection atom into a degenerate torsion.
                        atom_downstream_of_conn[i, 0] = i_conn_atom
                        continue
                    for j in range(self.n_atoms):
                        atom_downstream_of_conn[i, j] = self.atom_to_idx[
                            mc_ats[j] if j < len(mc_ats) else mc_ats[-1]
                        ]
            elif self.connections[i].name == "up":
                # walk up through the mainchain atoms untill we
                # hit the first mainchain atom and then report all
                # the other atoms downstream of the connection as the first
                mc_ats = self.properties.polymer.mainchain_atoms
                if mc_ats is None:
                    # weird case? The user has a "down" connection but no
                    # atoms are part of the mainchain?
                    atom_downstream_of_conn[i, :] = i_conn_atom
                else:
                    assert mc_ats[-1] == self.connections[i].atom
                    if len(mc_ats) == 1:
                        atom_downstream_of_conn[i, 0] = i_conn_atom
                        continue
                    for j in range(self.n_atoms):
                        atom_downstream_of_conn[i, j] = self.atom_to_idx[
                            (
                                mc_ats[len(mc_ats) - j - 1]
                                if j < len(mc_ats)
                                else mc_ats[0]
                            )
                        ]

            else:
                # we walk backwards through the parents of the the
                # connection atom; when we hit the root atom, just
                # keep going -- report all the other atoms downstream of
                # the connection as the root atom.
                parent = self.connections[i].atom
                for j in range(self.n_atoms):
                    atom_downstream_of_conn[i, j] = self.atom_to_idx[parent]
                    atom_index = atom_downstream_of_conn[i, j]
                    atom = parent
                    if self.icoors[atom_index].name == atom:
                        parent = self.icoors[atom_index].parent
                    else:
                        parent = next(x.parent for x in self.icoors if x.name == atom)
        return atom_downstream_of_conn

    # ---- icoors --------------------------------------------------------

    @property
    def n_icoors(self) -> int:
        return len(self.icoors)

    icoors_index: Mapping[str, IcoorIndex] = attr.ib()

    @icoors_index.default
    def _setup_icoors_index(self) -> Mapping[str, IcoorIndex]:
        return {icoor.name: i for i, icoor in enumerate(self.icoors)}

    at_to_icoor_ind: np.ndarray = attr.ib()

    @at_to_icoor_ind.default
    def _setup_at_to_icoor_ind(self) -> np.ndarray:
        return np.array(
            [self.icoors_index[at.name] for at in self.atoms], dtype=np.int32
        )

    icoors_ancestors: np.ndarray = attr.ib()

    @icoors_ancestors.default
    def _setup_icoors_ancestors(self) -> np.ndarray:
        icoors_ancestors = np.full((len(self.icoors), 3), -1, dtype=np.int32)
        for i in range(len(self.icoors)):
            for j in range(3):
                if j == 0:
                    at = self.icoors[i].parent
                elif j == 1:
                    at = self.icoors[i].grand_parent
                else:
                    at = self.icoors[i].great_grand_parent
                icoors_ancestors[i, j] = self.icoors_index[at]
        return icoors_ancestors

    icoors_geom: np.ndarray = attr.ib()

    @icoors_geom.default
    def _setup_icoors_geom(self) -> np.ndarray:
        icoors_geom = np.zeros((len(self.icoors), 3), dtype=np.float64)
        for i in range(len(self.icoors)):
            icoors_geom[i, 0] = self.icoors[i].phi
            icoors_geom[i, 1] = self.icoors[i].theta
            icoors_geom[i, 2] = self.icoors[i].d
        return icoors_geom

    ideal_coords: np.ndarray = attr.ib()

    @ideal_coords.default
    def _compute_ideal_coords(self) -> np.ndarray:
        return build_coords_from_icoors(self.icoors_ancestors, self.icoors_geom)

    # ---- jump connection atom -----------------------------------------

    default_jump_connection_atom_index: int = attr.ib()

    @default_jump_connection_atom_index.default
    def _get_default_jump_connection_atom_index(self) -> int:
        return self.atom_to_idx[self.default_jump_connection_atom]

    # ---- nice repr -----------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover -- debugging helper
        return f"RefinedResidueType(name={self.name!r}, n_atoms={self.n_atoms})"

    def _repr_pretty_(self, p, cycle):  # pragma: no cover
        p.text(f"RefinedResidueType(name={self.name},...)")


# --------------------------------------------------------------------- #
# From RawResidueType -> RefinedResidueType                              #
# --------------------------------------------------------------------- #


def _refined_from_raw(raw: RawResidueType) -> RefinedResidueType:
    """Build a ``RefinedResidueType`` from the raw fields of a ``RawResidueType``.

    All the derived fields have defaults so we just copy the raw fields.
    """
    return RefinedResidueType(
        **{
            field.name: getattr(raw, field.name)
            for field in attr.fields(RawResidueType)
        }
    )


# --------------------------------------------------------------------- #
# ResidueTypeSet                                                         #
# --------------------------------------------------------------------- #


@attr.s(auto_attribs=True)
class ResidueTypeSet:
    """A set of residue types grouped by their ``name3`` code."""

    residue_types: Sequence[RefinedResidueType]
    restype_map: Mapping[ResName3, Sequence[RefinedResidueType]]
    chem_db: typing.Any  # PatchedChemicalDatabase (avoid cyclic import)

    _default: typing.ClassVar[Optional["ResidueTypeSet"]] = None

    @classmethod
    def get_default(cls) -> "ResidueTypeSet":
        """Return the residue-type set built from the default chemical DB."""
        if cls._default is None:
            # Import lazily to avoid a circular import at module load time.
            from tmol_jax.chemical.patched_chemdb import default_patched_chemdb

            cls._default = cls.from_database(default_patched_chemdb())
        return cls._default

    @classmethod
    def from_database(cls, chemical_db) -> "ResidueTypeSet":
        residue_types = [_refined_from_raw(r) for r in chemical_db.residues]
        restype_map: Dict[ResName3, List[RefinedResidueType]] = {}
        for rt in residue_types:
            restype_map.setdefault(rt.name3, []).append(rt)
        return cls(
            residue_types=residue_types,
            restype_map=restype_map,
            chem_db=chemical_db,
        )

    @classmethod
    def from_restype_list(
        cls, chemical_db, restypes: List[RefinedResidueType]
    ) -> "ResidueTypeSet":
        restype_map: Dict[ResName3, List[RefinedResidueType]] = {}
        for rt in restypes:
            restype_map.setdefault(rt.name3, []).append(rt)
        return cls(
            residue_types=restypes,
            restype_map=restype_map,
            chem_db=chemical_db,
        )


__all__ = [
    "AtomIndex",
    "ConnectionIndex",
    "BondCount",
    "UnresolvedAtomID",
    "uaid_t",
    "ResName3",
    "IcoorIndex",
    "RefinedResidueType",
    "ResidueTypeSet",
]
