import attr
import numpy
import torch

import sparse
import scipy.sparse.csgraph as csgraph

# from tmol.system.restypes import RefinedResidueType
from tmol.system.pose import PackedBlockTypes, Poses
from tmol.score.EnergyTerm import EnergyTerm


@attr.s(auto_attribs=True)
class BondDependentTerm(EnergyTerm):
    device: torch.device

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(BondDependentTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "bond_separation"):
            assert hasattr(packed_block_types, "max_n_interblock_bonds")
            assert hasattr(packed_block_types, "n_interblock_bonds")
            assert hasattr(packed_block_types, "atoms_for_interblock_bonds")
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
        for i, rt in enumerate(packed_block_types.active_residues):
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
            len(rt.connections) for rt in packed_block_types.active_residues
        ]
        max_n_interblock_bonds = max(n_interblock_bonds)
        n_interblock_bonds = numpy.array(n_interblock_bonds, dtype=numpy.int32)
        atoms_for_interblock_bonds = numpy.full(
            (packed_block_types.n_types, max_n_interblock_bonds), -1, dtype=numpy.int32
        )
        for i, rt in enumerate(packed_block_types.active_residues):
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
            assert hasattr(systems, "inter_block_bondsep_t")
            return

        n_systems = systems.coords.shape[0]
        max_n_blocks = systems.coords.shape[1]

        min_block_bondsep = numpy.min(systems.inter_block_bondsep, axis=4)
        min_block_bondsep = numpy.min(min_block_bondsep, axis=3)

        min_block_bondsep = torch.tensor(min_block_bondsep, device=self.device)
        inter_block_bondsep_t = torch.tensor(
            systems.inter_block_bondsep, device=self.device
        )

        setattr(systems, "min_block_bondsep", min_block_bondsep)
        setattr(systems, "inter_block_bondsep_t", inter_block_bondsep_t)