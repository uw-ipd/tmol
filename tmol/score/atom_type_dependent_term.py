import attr
import numpy
import torch
import pandas

from tmol.database import ParameterDatabase
from tmol.database.chemical import ChemicalDatabase
from .chemical_database import AtomTypeParamResolver
from tmol.system.pose import PackedBlockTypes
from tmol.system.restypes import RefinedResidueType
from tmol.score.EnergyTerm import EnergyTerm

from tmol.types.array import NDArray


class AtomTypeDependentTerm(EnergyTerm):
    atom_type_resolver: AtomTypeParamResolver  # = attr.ib()
    atom_type_index: pandas.Index  # = attr.ib()
    np_is_hydrogen: NDArray[bool]  # = attr.ib()
    np_is_heavyatom: NDArray[bool]  # = attr.ib()
    device: torch.device  # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        atom_type_resolver = AtomTypeParamResolver.from_database(
            param_db.chemical, device=device
        )
        super(AtomTypeDependentTerm, self).__init__(param_db=param_db, device=device)
        self.atom_type_resolver = atom_type_resolver
        self.atom_type_index = atom_type_resolver.index
        self.np_is_hydrogen = atom_type_resolver.params.is_hydrogen.cpu().numpy()
        self.np_is_heavyatom = numpy.logical_not(self.np_is_hydrogen)
        self.device = device

    def setup_block_type(self, block_type: RefinedResidueType):
        super(AtomTypeDependentTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "atom_types"):
            assert hasattr(block_type, "heavy_atom_inds")
            return
        atom_types = self.atom_type_index.get_indexer(
            [x.atom_type for x in block_type.atoms]
        )
        heavy_inds = numpy.nonzero(self.np_is_heavyatom[atom_types])[0]
        setattr(block_type, "atom_types", atom_types)
        setattr(block_type, "heavy_atom_inds", heavy_inds)

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

        for i, restype in enumerate(packed_block_types.active_block_types):
            atom_types[
                i, : packed_block_types.n_atoms[i]
            ] = self.atom_type_index.get_indexer([x.atom_type for x in restype.atoms])

        heavy_atom_inds = []
        for restype in packed_block_types.active_block_types:
            rt_heavy = [
                j
                for j, atype_ind in enumerate(
                    self.atom_type_resolver.index.get_indexer(
                        [restype.atoms[j].atom_type for j in range(len(restype.atoms))]
                    )
                )
                if not self.atom_type_resolver.params.is_hydrogen[atype_ind]
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
