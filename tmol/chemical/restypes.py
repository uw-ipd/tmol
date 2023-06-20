from frozendict import frozendict
from toolz.curried import concat, map, compose, groupby
import typing
from typing import Mapping, Optional, NewType, Tuple, Sequence, List
import attr
import cattr

import numpy
import scipy.sparse
import scipy.sparse.csgraph as csgraph

from tmol.database import ParameterDatabase
from tmol.database.chemical import RawResidueType, ChemicalDatabase

from tmol.chemical.constants import MAX_SIG_BOND_SEPARATION
from tmol.chemical.ideal_coords import build_coords_from_icoors
from tmol.chemical.all_bonds import bonds_and_bond_ranges
from tmol.types.functional import validate_args


AtomIndex = NewType("AtomIndex", int)
ConnectionIndex = NewType("ConnectionIndex", int)
BondCount = NewType("BondCount", int)

# perhaps deserving of its own file
UnresolvedAtomID = Tuple[AtomIndex, ConnectionIndex, BondCount]
uaid_t = numpy.dtype(
    [
        ("atom_id", numpy.int32),
        ("conn_id", numpy.int32),
        ("n_bonds_from_conn", numpy.int32),
    ]
)

ResName3 = typing.NewType("ResName3", str)
IcoorIndex = NewType("AtomIndex", int)


@attr.s
class RefinedResidueType(RawResidueType):
    @property
    def n_atoms(self):
        return len(self.atoms)

    atom_to_idx: Mapping[str, AtomIndex] = attr.ib()

    @atom_to_idx.default
    def _setup_atom_to_idx(self):
        return frozendict((a.name, i) for i, a in enumerate(self.atoms))

    coord_dtype: numpy.dtype = attr.ib()

    @coord_dtype.default
    def _setup_coord_dtype(self):
        return numpy.dtype([(a.name, float, 3) for a in self.atoms])

    bond_indices: numpy.ndarray = attr.ib()

    @bond_indices.default
    def _setup_bond_indices(self):
        bondi = compose(list, sorted, set, concat)(
            [(ai, bi), (bi, ai)]
            for ai, bi in map(map(self.atom_to_idx.get), self.bonds)
        )

        bond_array = numpy.array(bondi, dtype=numpy.int32)
        bond_array.flags.writeable = False
        return bond_array

    # The index of the atom for a given inter-residue connection point
    connection_to_idx: Mapping[str, AtomIndex] = attr.ib()

    @connection_to_idx.default
    def _setup_connection_to_idx(self):
        return frozendict((c.name, self.atom_to_idx[c.atom]) for c in self.connections)

    connection_to_cidx: Mapping[Optional[str], ConnectionIndex] = attr.ib()

    @connection_to_cidx.default
    def _setup_connection_to_cidx(self):
        centries = [(None, -1)] + [(c.name, i) for i, c in enumerate(self.connections)]
        return frozendict(centries)

    ordered_connection_atoms: numpy.ndarray = attr.ib()

    @ordered_connection_atoms.default
    def _setup_ordered_connections(self):
        return numpy.array(
            [self.atom_to_idx[c.atom] for c in self.connections], dtype=numpy.int32
        )

    # The set of "all-bonds" includes both inter- and intra- block chemical bonds
    # Each all-bond (think of "all" here as an adjective like "inter" or "intra")
    # is described as a pair (atom-ind1, unresolved-atom id) where the
    # unresolved-atom id is a tuple (intra-block-atom-ind, conn-id)
    # following the same rules for undresolved-atom ids for describing
    # torsions, except that the "bond sep from conn" integer is always 0
    # because the chemical bond is always defined as to the connection
    # atom on the other residue, so it is omitted from the all-bonds info.
    # That is, the "intra-block-atom-ind" is >= 0 if the atom to which atom_ind1
    # is chemically bound and is in the same block as it, and is -1 otherwise;
    # conn-id is >= 0 and represents the index of the connection on *this*
    # block that connects atom_ind1 to its partner when the partner is
    # on a different block and is -1 otherwise.
    all_bonds: numpy.ndarray = attr.ib()

    # NOTE: this also creates self.atom_all_bond_ranges
    @all_bonds.default
    def _setup_all_bonds(self):
        all_bonds, atom_all_bond_ranges = bonds_and_bond_ranges(
            self.n_atoms, self.bond_indices, self.ordered_connection_atoms
        )
        self.atom_all_bond_ranges = atom_all_bond_ranges
        return all_bonds

    down_connection_ind: int = attr.ib()

    @down_connection_ind.default
    def _setup_down_connection_ind(self):
        if "down" in self.connection_to_cidx:
            return self.connection_to_cidx["down"]
        else:
            return -1

    up_connection_ind: int = attr.ib()

    @up_connection_ind.default
    def _setup_up_connection_ind(self):
        if "up" in self.connection_to_cidx:
            return self.connection_to_cidx["up"]
        else:
            return -1

    def _repr_pretty_(self, p, cycle):
        p.text(f"RefinedResidueType(name={self.name},...)")

    torsion_to_uaids: Mapping[str, Tuple[UnresolvedAtomID]] = attr.ib()

    @torsion_to_uaids.default
    def _setup_torsion_to_uaids(self):
        torsion_to_uaids = {}
        for tor in self.torsions:
            ats = []
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
            torsion_to_uaids[tor.name] = ats
        return frozendict(torsion_to_uaids)

    ordered_torsions: numpy.ndarray = attr.ib()

    @ordered_torsions.default
    def _setup_ordered_torsions(self):
        ordered_torsions = numpy.full((len(self.torsions), 4, 3), -1, dtype=numpy.int32)
        for i, tor in enumerate(self.torsions):
            for j in range(4):
                ordered_torsions[i, j] = numpy.array(
                    self.torsion_to_uaids[tor.name][j], dtype=numpy.int32
                )
        return ordered_torsions

    path_distance: numpy.ndarray = attr.ib()

    @path_distance.default
    def _setup_path_distance(self):
        bonds_sparse = scipy.sparse.coo_matrix(
            (
                numpy.full(self.bond_indices.shape[0], True),
                (self.bond_indices[:, 0], self.bond_indices[:, 1]),
            ),
            shape=(self.n_atoms, self.n_atoms),
        )
        path_distance = csgraph.dijkstra(
            bonds_sparse, directed=False, unweighted=True, limit=MAX_SIG_BOND_SEPARATION
        )
        path_distance[path_distance == numpy.inf] = MAX_SIG_BOND_SEPARATION
        return path_distance

    atom_downstream_of_conn: numpy.ndarray = attr.ib()

    @atom_downstream_of_conn.default
    def _setup_atom_downstream_of_conn(self):
        n_conns = len(self.connections)
        atom_downstream_of_conn = numpy.full(
            (n_conns, self.n_atoms), -1, dtype=numpy.int32
        )
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
                    for j in range(self.n_atoms):
                        atom_downstream_of_conn[i, j] = self.atom_to_idx[
                            mc_ats[j] if j < len(mc_ats) else mc_ats[-1]
                        ]
            elif self.connections[i].name == "up":
                # walk up through the mainchain atoms untill we
                # hit the first mainchain atom and then report all
                # the other atoms downstream of the connection as the first
                assert mc_ats[-1] == self.connections[i].atom
                mc_ats = self.properties.polymer.mainchain_atoms
                if mc_ats is None:
                    # weird case? The user has a "down" connection but no
                    # atoms are part of the mainchain?
                    atom_downstream_of_conn[i, :] = i_conn_atom
                else:
                    for j in range(self.n_atoms):
                        atom_downstream_of_conn[i, j] = self.atom_to_idx[
                            mc_ats[len(mc_ats) - j - 1]
                            if j < len(mc_ats)
                            else mc_ats[0]
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

    @property
    def n_icoors(self):
        return len(self.icoors)

    icoors_index: Mapping[str, IcoorIndex] = attr.ib()

    @icoors_index.default
    def _setup_icoors_index(self):
        return {icoor.name: i for i, icoor in enumerate(self.icoors)}

    at_to_icoor_ind: numpy.array = attr.ib()

    @at_to_icoor_ind.default
    def _setup_at_to_icoor_ind(self):
        return numpy.array(
            [self.icoors_index[at.name] for at in self.atoms], dtype=numpy.int32
        )

    icoors_ancestors: numpy.ndarray = attr.ib()

    @icoors_ancestors.default
    def _setup_icoors_ancestors(self):
        icoors_ancestors = numpy.full((len(self.icoors), 3), -1, dtype=numpy.int32)
        for i in range(len(self.icoors)):
            for j in range(3):
                at = (
                    self.icoors[i].parent
                    if j == 0
                    else (
                        self.icoors[i].grand_parent
                        if j == 1
                        else self.icoors[i].great_grand_parent
                    )
                )
                icoors_ancestors[i, j] = self.icoors_index[at]

        return icoors_ancestors

    icoors_geom: numpy.ndarray = attr.ib()

    @icoors_geom.default
    def _setup_icoors_geom(self):
        icoors_geom = numpy.zeros((len(self.icoors), 3), dtype=numpy.float64)
        for i in range(len(self.icoors)):
            for j in range(3):
                icoors_geom[i, j] = (
                    self.icoors[i].phi
                    if j == 0
                    else (self.icoors[i].theta if j == 1 else self.icoors[i].d)
                )
        return icoors_geom

    ideal_coords: numpy.ndarray = attr.ib()

    @ideal_coords.default
    def compute_ideal_coords(self):
        return build_coords_from_icoors(self.icoors_ancestors, self.icoors_geom)


@attr.s(auto_attribs=True)
class ResidueTypeSet:
    __default = None

    @classmethod
    def get_default(cls) -> "ResidueTypeSet":
        """Load and return the residue type set constructed from the default param db"""
        if cls.__default is None:
            cls.__default = cls.from_database(ParameterDatabase.get_default().chemical)
        return cls.__default

    @classmethod
    def from_database(cls, chemical_db: ChemicalDatabase):
        residue_types = [
            cattr.structure(cattr.unstructure(r), RefinedResidueType)
            for r in chemical_db.residues
        ]
        restype_map = groupby(lambda restype: restype.name3, residue_types)
        return cls(residue_types=residue_types, restype_map=restype_map)

    residue_types: Sequence[RefinedResidueType]
    restype_map: Mapping[ResName3, Sequence[RefinedResidueType]]


@attr.s(slots=True, frozen=True)
class Residue:
    residue_type: RefinedResidueType = attr.ib()
    coords: numpy.ndarray = attr.ib()

    @coords.default
    def _coord_buffer(self):
        return numpy.full((self.residue_type.n_atoms, 3), numpy.nan, dtype=float)

    @property
    def atom_coords(self) -> numpy.ndarray:
        return self.coords.reshape(-1).view(self.residue_type.coord_dtype)

    def attach_to(self, coord_buffer):
        assert coord_buffer.shape == self.coords.shape
        assert coord_buffer.dtype == self.coords.dtype

        coord_buffer[:] = self.coords

        return attr.evolve(self, coords=coord_buffer)

    def _repr_pretty_(self, p, cycle):
        p.text("Residue")
        with p.group(1, "(", ")"):
            p.text("residue_type=")
            p.pretty(self.residue_type)
            p.text(", coords=")
            p.break_()
            p.pretty(self.coords)


@validate_args
def find_simple_polymeric_connections(
    res: List[Residue],
) -> List[Tuple[int, str, int, str]]:
    """
    return a list of (int,str,int,str) quadrouples that say residue
    i is connected to residue i+1 from it's "up" connection to
    residue i+1's "down" connection and vice versa for all i"""

    residue_connections = []
    for i, j in zip(range(len(res) - 1), range(1, len(res))):
        if (
            "up" in res[i].residue_type.connection_to_idx
            and "down" in res[j].residue_type.connection_to_idx
        ):
            residue_connections.extend(
                [(i, "up", i + 1, "down"), (i + 1, "down", i, "up")]
            )

    return residue_connections


@validate_args
def find_disulfide_connections(
    res: List[Residue],
) -> List[Tuple[int, str, int, str]]:
    residue_connections = []

    cystines = [
        (ind, cys) for ind, cys in enumerate(res) if cys.residue_type.name == "CYD"
    ]
    for i, cys1 in cystines:
        for j, cys2 in cystines:
            if i < j:
                sg_index = cys1.residue_type.atom_to_idx["SG"]
                sg1 = cys1.coords[sg_index]
                sg2 = cys2.coords[sg_index]

                dist = numpy.linalg.norm(sg1 - sg2)

                if numpy.isclose(dist, 2.02, atol=0.5):
                    residue_connections.extend(
                        [(i, "dslf", j, "dslf"), (j, "dslf", i, "dslf")]
                    )
    return residue_connections
