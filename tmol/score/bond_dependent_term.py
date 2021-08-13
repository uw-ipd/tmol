import attr
import numpy
import torch

import sparse
import scipy.sparse.csgraph as csgraph

from tmol.database import ParameterDatabase
from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import Poses
from tmol.score.EnergyTerm import EnergyTerm
from tmol.score.bonded_atom import IndexedBonds


# @attr.s(auto_attribs=True)
class BondDependentTerm(EnergyTerm):
    device: torch.device

    def __init__(self, param_db: ParameterDatabase, device: torch.device, **kwargs):
        super(BondDependentTerm, self).__init__(param_db=param_db, device=device)
        self.device = device

    def setup_block_type(self, block_type: RefinedResidueType):
        super(BondDependentTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "intrares_indexed_bonds"):
            return

        bonds = numpy.zeros((block_type.bond_indices.shape[0], 3), dtype=numpy.int32)
        bonds[:, 1:] = block_type.bond_indices.astype(numpy.int32)
        ib = IndexedBonds.from_bonds(bonds, minlength=block_type.n_atoms)
        setattr(block_type, "intrares_indexed_bonds", ib)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(BondDependentTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "bond_separation"):
            assert hasattr(packed_block_types, "max_n_interblock_bonds")
            assert hasattr(packed_block_types, "n_interblock_bonds")
            assert hasattr(packed_block_types, "atoms_for_interblock_bonds")
            assert hasattr(packed_block_types, "intrares_indexed_bonds")
            return
        MAX_SEPARATION = 6
        bond_separation = numpy.full(
            (
                packed_block_types.n_types,
                packed_block_types.max_n_atoms,
                packed_block_types.max_n_atoms,
            ),
            MAX_SEPARATION,
            dtype=numpy.int32,
        )
        for i, rt in enumerate(packed_block_types.active_block_types):
            i_nats = packed_block_types.n_atoms[i]
            # rt_bonds = numpy.zeros((i_nats, i_nats)
            rt_bonds_sparse = sparse.COO(
                rt.bond_indices.T,
                data=numpy.full(len(rt.bond_indices), True),
                shape=(len(rt.atoms), len(rt.atoms)),
                cache=True,
            )
            rt_bond_closure = csgraph.dijkstra(
                rt_bonds_sparse, directed=False, unweighted=True, limit=MAX_SEPARATION
            )
            rt_bond_closure[rt_bond_closure == numpy.inf] = MAX_SEPARATION
            bond_separation[i, :i_nats, :i_nats] = rt_bond_closure
        bond_separation = torch.tensor(bond_separation, device=self.device)

        n_interblock_bonds = [
            len(rt.connections) for rt in packed_block_types.active_block_types
        ]
        max_n_interblock_bonds = max(n_interblock_bonds)
        n_interblock_bonds = numpy.array(n_interblock_bonds, dtype=numpy.int32)
        atoms_for_interblock_bonds = numpy.full(
            (packed_block_types.n_types, max_n_interblock_bonds), -1, dtype=numpy.int32
        )
        for i, rt in enumerate(packed_block_types.active_block_types):
            i_n_intres_bonds = n_interblock_bonds[i]
            if i_n_intres_bonds == 0:
                continue
            cnx_atoms = [rt.atom_to_idx[conn.atom] for conn in rt.connections]
            atoms_for_interblock_bonds[i, :i_n_intres_bonds] = cnx_atoms

        n_interblock_bonds = torch.tensor(n_interblock_bonds, device=self.device)
        atoms_for_interblock_bonds = torch.tensor(
            atoms_for_interblock_bonds, device=self.device
        )

        setattr(packed_block_types, "bond_separation", bond_separation)
        setattr(packed_block_types, "max_n_interblock_bonds", max_n_interblock_bonds)
        setattr(packed_block_types, "n_interblock_bonds", n_interblock_bonds)
        setattr(
            packed_block_types, "atoms_for_interblock_bonds", atoms_for_interblock_bonds
        )

    def setup_poses(self, systems: Poses):
        super(BondDependentTerm, self).setup_poses(systems)

        if hasattr(systems, "min_block_bondsep"):
            return

        min_block_bondsep, _ = torch.min(systems.inter_block_bondsep, dim=4)
        min_block_bondsep, _ = torch.min(min_block_bondsep, dim=3)

        setattr(systems, "min_block_bondsep", min_block_bondsep)
