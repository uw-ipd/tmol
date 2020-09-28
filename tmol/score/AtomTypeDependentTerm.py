import attr
import numpy
import torch
import pandas

from tmol.database.chemical import ChemicalDatabase
from .chemical_database import AtomTypeParamResolver
from tmol.system.pose import PackedBlockTypes
from tmol.score.EnergyTerm import EnergyTerm


@attr.s(auto_attribs=True)
class AtomTypeDependentTerm(EnergyTerm):
    resolver: AtomTypeParamResolver
    atom_type_index: pandas.Index
    device: torch.device

    @classmethod
    def from_database(cls, chemical_database: ChemicalDatabase, device: torch.device):
        return cls.from_param_resolver(
            AtomTypeParamResolver.from_database(chemical_database, device=device),
            device=device,
        )

    @classmethod
    def from_param_resolver(cls, resolver: AtomTypeParamResolver, device: torch.device):
        return cls(resolver=resolver, atom_type_index=resolver.index, device=device)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(AtomTypeDependentTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "atom_types"):
            assert hasattr(packed_block_types, "n_heavy_atoms")
            assert hasattr(packed_block_types, "heavy_atom_inds")
            return
        atom_types = numpy.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms),
            -1,
            dtype=numpy.int32,
        )

        for i, restype in enumerate(packed_block_types.active_residues):
            atom_types[
                i, : packed_block_types.n_atoms[i]
            ] = self.atom_type_index.get_indexer([x.atom_type for x in restype.atoms])

        heavy_atom_inds = []
        for restype in packed_block_types.active_residues:
            rt_heavy = [
                j
                for j, atype_ind in enumerate(
                    self.resolver.index.get_indexer(
                        [restype.atoms[j].atom_type for j in range(len(restype.atoms))]
                    )
                )
                if not self.resolver.params.is_hydrogen[atype_ind]
            ]
            heavy_atom_inds.append(rt_heavy)

        n_heavy_atoms = numpy.array(
            [len(heavy_inds) for heavy_inds in heavy_atom_inds], dtype=numpy.int32
        )
        max_n_heavy = numpy.max(n_heavy_atoms)

        heavy_atom_inds_t = torch.full(
            (packed_block_types.n_types, max_n_heavy), -1, dtype=torch.int32
        )
        for i, inds in enumerate(heavy_atom_inds):
            heavy_atom_inds_t[i, : len(inds)] = torch.tensor(inds, dtype=torch.int32)

        atom_types = torch.tensor(atom_types, device=self.device)
        heavy_atom_inds_t = heavy_atom_inds_t.to(self.device)
        n_heavy_atoms = torch.tensor(
            n_heavy_atoms, dtype=torch.int32, device=self.device
        )

        setattr(packed_block_types, "atom_types", atom_types)
        setattr(packed_block_types, "n_heavy_atoms", n_heavy_atoms)
        setattr(packed_block_types, "heavy_atom_inds", heavy_atom_inds_t)
