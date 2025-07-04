from frozendict import frozendict
from toolz.curried import concat, map, compose, groupby
import typing
from typing import Mapping, Optional, NewType, Tuple, Sequence, List, Set, Union
import attr
import cattr

import numpy
import scipy.sparse
import scipy.sparse.csgraph as csgraph

from tmol.database import ParameterDatabase
from tmol.database.chemical import RawResidueType
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase

from tmol.chemical.constants import MAX_SIG_BOND_SEPARATION
from tmol.chemical.constants import MAX_PATHS_FROM_CONNECTION
from tmol.chemical.ideal_coords import build_coords_from_icoors
from tmol.chemical.all_bonds import bonds_and_bond_ranges


AtomIndex = NewType("AtomIndex", int)
ConnectionIndex = NewType("ConnectionIndex", int)
BondCount = NewType("BondCount", int)

# As of cattr 24.1.0, more types must be explicitly registered in order to
# use cattr.structure. We use that here
cattr.register_structure_hook(numpy.dtype, lambda d, _: numpy.dtype(d))
cattr.register_structure_hook(numpy.ndarray, lambda d, _: numpy.array(d))


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


def three2one(three: str) -> Union[str, None]:
    """Return the one-letter amino acid code given its three letter code,
    or None if not a valid three-letter code
    """
    # 'static'
    if not hasattr(three2one, "_mapping"):
        three2one._mapping = {
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
        }
    if three in three2one._mapping:
        return three2one._mapping[three]
    return None


def one2three(one: str) -> Union[str, None]:
    """Return the three-letter amino acid code given its one-letter code,
    or None if not a valid one-letter code.
    """
    # 'static'
    if not hasattr(one2three, "_mapping"):
        one2three._mapping = {
            "A": "ALA",
            "C": "CYS",
            "D": "ASP",
            "E": "GLU",
            "F": "PHE",
            "G": "GLY",
            "H": "HIS",
            "I": "ILE",
            "K": "LYS",
            "L": "LEU",
            "M": "MET",
            "N": "ASN",
            "P": "PRO",
            "Q": "GLN",
            "R": "ARG",
            "S": "SER",
            "T": "THR",
            "V": "VAL",
            "W": "TRP",
            "Y": "TYR",
        }
    if one in one2three._mapping:
        return one2three._mapping[one]
    return None


@attr.s
class RefinedResidueType(RawResidueType):
    @property
    def n_atoms(self):
        return len(self.atoms)

    atom_names_set: Set[str] = attr.ib()

    @atom_names_set.default
    def _atom_names_set(self):
        return set([a.name for a in self.atoms])

    atom_to_idx: Mapping[str, AtomIndex] = attr.ib()

    @atom_to_idx.default
    def _setup_atom_to_idx(self):
        return frozendict((a.name, i) for i, a in enumerate(self.atoms))

    aliases_map: Mapping[str, str] = attr.ib()

    @aliases_map.default
    def _setup_atom_aliases_mapping(self):
        # make sure no alias is for an existing atom
        # and that there are no name colisions among
        # the aliases
        aliases = set([])
        for a in self.atom_aliases:
            assert a.alt_name not in self.atom_to_idx
            assert a.alt_name not in aliases
            aliases.add(a.alt_name)
        return frozendict((a.alt_name, a.name) for a in self.atom_aliases)

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

    @property
    def n_conn(self):
        return len(self.connections)

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

    @property
    def n_torsions(self):
        return self.ordered_torsions.shape[0]

    is_torsion_mc: numpy.ndarray = attr.ib()

    @is_torsion_mc.default
    def _setup_is_torsion_mc(self):
        # A torsion is a "main chain" torsion if all of its atoms are
        # listed as main chain atoms, or if they are listed as parts of
        # other residues. E.g., omega is a main chain torsion because
        # CA and C are both main chain atoms, and the other two atoms
        # belong to the next residue; however, chi 1 is not main chain
        # because, even though N and CA are listed as main chain atoms,
        # CB and CG are not.
        def all_torsion_atoms_are_mainchain(tor_ind):
            if not self.properties.polymer.is_polymer:
                return False
            # check that all atoms in the torsion are either part of the
            # mainchain or are atoms from other residues
            for i in range(4):
                at_i = self.ordered_torsions[tor_ind, i, 0]
                if at_i == -1:
                    continue
                atname = self.atoms[at_i].name
                if atname not in self.properties.polymer.mainchain_atoms:
                    return False
            return True

        return numpy.array(
            [all_torsion_atoms_are_mainchain(tor) for tor in range(self.n_torsions)],
            dtype=bool,
        )

    # torsions are either "main chain" or they are "side chain"; if a residue is not
    # polymeric, then all of its named torsions are side chain.
    mc_torsions: numpy.ndarray = attr.ib()

    @mc_torsions.default
    def _setup_mc_torsions(self):
        return numpy.nonzero(self.is_torsion_mc)[0].astype(numpy.int32)

    @property
    def n_mc_torsions(self):
        return len(self.mc_torsions)

    sc_torsions: numpy.ndarray = attr.ib()

    @sc_torsions.default
    def _setup_sc_torsions(self):
        return numpy.nonzero(numpy.logical_not(self.is_torsion_mc))[0].astype(
            numpy.int32
        )

    @property
    def n_sc_torsions(self):
        return len(self.sc_torsions)

    which_mcsc_torsion: numpy.ndarray = attr.ib()

    @which_mcsc_torsion.default
    def _setup_which_mcsc_torsion(self):
        which_mcsc_torsion = numpy.full(self.n_torsions, -1, dtype=numpy.int32)
        for i in range(self.n_mc_torsions):
            which_mcsc_torsion[self.mc_torsions[i]] = i
        for i in range(self.n_sc_torsions):
            which_mcsc_torsion[self.sc_torsions[i]] = i
        return which_mcsc_torsion

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
        return path_distance.astype(numpy.int32)

    atom_paths_from_conn: numpy.ndarray = attr.ib()

    @atom_paths_from_conn.default
    def _setup_atom_paths_from_conn(self):
        n_conns = len(self.connections)

        atom_paths = numpy.full(
            (n_conns, MAX_PATHS_FROM_CONNECTION, 3), -1, dtype=numpy.int32
        )

        if n_conns == 0:
            return atom_paths

        # Create a numpy array with the paths coming from a connection.
        # The first entry will be the immediate atom, followed by the
        # 3 paths coming from that atom, followed by the 3 coming out
        # of each of those in turn. If a path doesn't exist, it is
        # filled with -1s to ensure deterministic indexing of the paths.
        def get_paths_length_3(connection):
            paths = numpy.full((MAX_PATHS_FROM_CONNECTION, 3), -1, dtype=numpy.int32)
            # create a convenient datastructure for following connections
            bondmap = {-1: []}
            for bond in self.bond_indices:
                if bond[0] not in bondmap.keys():
                    bondmap[bond[0]] = []
                bondmap[bond[0]].append(bond[1])

            atom0 = self.atom_to_idx[connection.atom]
            # Add the immediate atom
            paths[0] = (atom0, -1, -1)

            idx = 1
            # Add the 3 paths connecting to the immediate atom
            for atom1 in bondmap[atom0] + [-1] * (3 - len(bondmap[atom0])):
                if atom1 != -1:
                    paths[idx] = (atom0, atom1, -1)
                idx += 1

            # Add the 9 paths connecting to the 3 from the previous step
            for atom1 in bondmap[atom0] + [-1] * (3 - len(bondmap[atom0])):
                for atom2 in bondmap[atom1] + [-1] * (3 - len(bondmap[atom1])):
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
                mc_ats = self.properties.polymer.mainchain_atoms
                if mc_ats is None:
                    # weird case? The user has a "down" connection but no
                    # atoms are part of the mainchain?
                    atom_downstream_of_conn[i, :] = i_conn_atom
                else:
                    assert mc_ats[-1] == self.connections[i].atom
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

    @property
    def n_icoors(self):
        return len(self.icoors)

    icoors_index: Mapping[str, IcoorIndex] = attr.ib()

    @icoors_index.default
    def _setup_icoors_index(self):
        return {icoor.name: i for i, icoor in enumerate(self.icoors)}

    at_to_icoor_ind: numpy.ndarray = attr.ib()

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

    default_jump_connection_atom_index: int = attr.ib()

    @default_jump_connection_atom_index.default
    def get_default_jump_connection_atom_index(self):
        return self.atom_to_idx[self.default_jump_connection_atom]


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
    def from_database(cls, chemical_db: PatchedChemicalDatabase):
        residue_types = [
            cattr.structure(cattr.unstructure(r), RefinedResidueType)
            for r in chemical_db.residues
        ]
        restype_map = groupby(lambda restype: restype.name3, residue_types)
        return cls(
            residue_types=residue_types,
            restype_map=restype_map,
            chem_db=chemical_db,
        )

    @classmethod
    def from_restype_list(
        cls, chemical_db: PatchedChemicalDatabase, restypes: List[RefinedResidueType]
    ):
        restype_map = groupby(lambda restype: restype.name3, restypes)
        return cls(
            residue_types=restypes,
            restype_map=restype_map,
            chem_db=chemical_db,
        )

    residue_types: Sequence[RefinedResidueType]
    restype_map: Mapping[ResName3, Sequence[RefinedResidueType]]
    chem_db: PatchedChemicalDatabase
