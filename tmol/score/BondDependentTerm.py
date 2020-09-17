import attr
import numpy
import torch

import sparse
import scipy.sparse.csgraph as csgraph

from tmol.system.restypes import ResidueType
from tmol.system.pose import PackedResidueTypes


@attr.s(auto_attribs=True)
class BondDependentTerm:
    myint: int
    device: torch.device

    def setup_packed_restypes(self, packed_restypes: PackedResidueTypes):
        if hasattr(packed_restypes, "bond_separation"):
            assert hasattr(packed_restypes, "max_n_interres_bonds")
            assert hasattr(packed_restypes, "n_interres_bonds")
            assert hasattr(packed_restypes, "atoms_for_interres_bonds")
            return
        MAX_SEPARATION = 6
        bond_separation = numpy.full(
            (
                packed_restypes.n_types,
                packed_restypes.max_n_atoms,
                packed_restypes.max_n_atoms,
            ),
            MAX_SEPARATION,
            dtype=numpy.int32,
        )
        for i, rt in enumerate(packed_restypes.active_residues):
            i_nats = packed_restypes.n_atoms[i]
            # rt_bonds = numpy.zeros((i_nats, i_nats)
            rt_bonds_sparse = sparse.COO(
                rt.bond_indicies.T,
                data=numpy.full(len(rt.bond_indicies), True),
                shape=(len(rt.atoms), len(rt.atoms)),
                cache=True,
            )
            rt_bond_closure = csgraph.dijkstra(
                rt_bonds_sparse, directed=False, unweighted=True, limit=MAX_SEPARATION
            )
            bond_separation[i, :i_nats, :i_nats] = rt_bond_closure

        n_interres_bonds = [
            len(rt.connections) for rt in packed_restypes.active_residues
        ]
        max_n_interres_bonds = max(n_interres_bonds)
        n_interres_bonds = numpy.array(n_interres_bonds, dtype=numpy.int32)
        atoms_for_interres_bonds = numpy.full(
            (packed_restypes.n_types, max_n_interres_bonds), -1, dtype=numpy.int32
        )
        for i, rt in enumerate(packed_restypes.active_residues):
            i_n_intres_bonds = n_interres_bonds[i]
            if i_n_intres_bonds == 0:
                continue
            cnx_atoms = [rt.atom_to_idx[conn.atom] for conn in rt.connections]
            atoms_for_interres_bonds[i, :i_n_intres_bonds] = cnx_atoms

        n_interres_bonds = torch.tensor(n_interres_bonds, device=self.device)
        atoms_for_interres_bonds = torch.tensor(
            atoms_for_interres_bonds, device=self.device
        )

        setattr(packed_restypes, "bond_separation", bond_separation)
        setattr(packed_restypes, "max_n_interres_bonds", max_n_interres_bonds)
        setattr(packed_restypes, "n_interres_bonds", n_interres_bonds)
        setattr(packed_restypes, "atoms_for_interres_bonds", atoms_for_interres_bonds)
