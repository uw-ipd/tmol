import attr
import torch
import pandas

from tmol.database.chemical import ChemicalDatabase
from .chemical_database import AtomTypeParamResolver


@attr.s(auto_attribs=True)
class AtomTypeDependentTerm:
    atom_type_index: pandas.Index
    device: torch.device

    @classmethod
    def from_database(cls, chemical_database: ChemicalDatabase, device: torch.device):
        return cls.from_param_resolver(
            AtomTypeParamResolver.from_database(chemical_database, device=device)
        )

    @classmethod
    def from_param_resolver(cls, resolver: AtomTypeParamResolver):
        return cls(atom_type_index=resolver.index)

    def setup_packed_block_types(packed_block_types: PackedBlockTypes):
        if hasattr(packed_block_types, "atom_types"):
            return
        atom_types = numpy.full(
            (packed_block_types.n_types(), packed_restyes.max_n_atoms()),
            -1,
            dtype=numpy.int32,
        )

        for i, restype in enumerate(packed_block_types.active_restypes):
            atom_types[
                i, : packed_block_types.n_atoms[i]
            ] = atom_type_index.get_indexer([x.name for x in restype.atoms])

        atom_types = torch.tensor(atom_types, device=device)
        setattr(packed_block_types, "atom_types", atom_types)
