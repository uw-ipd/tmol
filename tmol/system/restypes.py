from frozendict import frozendict
from toolz.curried import concat, map, compose
from typing import Mapping, Optional, NewType, Tuple
import attr

import numpy
import sparse
import scipy.sparse.csgraph as csgraph

import tmol.database.chemical

AtomIndex = NewType("AtomIndex", int)
ConnectionIndex = NewType("ConnectionIndex", int)
BondCount = NewType("ConnectionIndex", int)
UnresolvedAtomID = Tuple[AtomIndex, ConnectionIndex, BondCount]


@attr.s
class RefinedResidueType(tmol.database.chemical.RawResidueType):
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

        bond_array = numpy.array(bondi)
        bond_array.flags.writeable = False
        return bond_array

    connection_to_idx: Mapping[str, AtomIndex] = attr.ib()

    @connection_to_idx.default
    def _setup_connection_to_idx(self):
        return frozendict((c.name, self.atom_to_idx[c.atom]) for c in self.connections)

    connection_to_cidx: Mapping[Optional[str], ConnectionIndex] = attr.ib()

    @connection_to_cidx.default
    def _setup_connection_to_cidx(self):
        centries = [(None, -1)] + [(c.name, i) for i, c in enumerate(self.connections)]
        return frozendict(centries)

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

    path_distance: numpy.ndarray = attr.ib()

    @path_distance.default
    def _setup_path_distance(self):
        MAX_SEPARATION = 6
        n_atoms = len(self.atoms)
        bonds_sparse = sparse.COO(
            self.bond_indices.T,
            data=numpy.full(len(self.bond_indices), True),
            shape=(n_atoms, n_atoms),
            cache=True,
        )
        path_distance = csgraph.dijkstra(
            bonds_sparse, directed=False, unweighted=True, limit=MAX_SEPARATION
        )
        path_distance[path_distance == numpy.inf] = MAX_SEPARATION
        return path_distance

    atom_downstream_of_conn: numpy.ndarray = attr.ib()

    @atom_downstream_of_conn.default
    def _setup_atom_downstream_of_conn(self):
        n_atoms = len(self.atoms)
        n_conns = len(self.connections)
        atom_downstream_of_conn = numpy.full((n_conns, n_atoms), -1, dtype=numpy.int32)
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
                    for j in range(n_atoms):
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
                    for j in range(n_atoms):
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
                for j in range(n_atoms):
                    atom_downstream_of_conn[i, j] = self.atom_to_idx[parent]
                    atom_index = atom_downstream_of_conn[i, j]
                    atom = parent
                    if self.icoors[atom_index].name == atom:
                        parent = self.icoors[atom_index].parent
                    else:
                        parent = next(x.parent for x in self.icoors if x.name == "atom")
        return atom_downstream_of_conn


@attr.s(slots=True, frozen=True)
class Residue:
    residue_type: RefinedResidueType = attr.ib()
    coords: numpy.ndarray = attr.ib()

    @coords.default
    def _coord_buffer(self):
        return numpy.full((len(self.residue_type.atoms), 3), numpy.nan, dtype=float)

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
