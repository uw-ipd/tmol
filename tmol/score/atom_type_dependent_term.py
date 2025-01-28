import numpy
import torch
import pandas

from tmol.database import ParameterDatabase
from .chemical_database import AtomTypeParamResolver
from tmol.chemical.restypes import RefinedResidueType
from tmol.score.energy_term import EnergyTerm

from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.pose.pose_stack import PoseStack

from tmol.types.array import NDArray

from collections import OrderedDict


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

    def get_atom_unique_id_name(self, block_name, atom_name):
        return "UNIQUE_ID:" + block_name + ":" + atom_name

    def get_atom_wildcard_id_name(self, atom_name):
        return "WILDCARD_ID:" + atom_name

    def _create_uniq_and_wildcard_names_for_bt(self, block_type):
        atom_names = [x.name for x in block_type.atoms]

        unique_ids = [
            self.get_atom_unique_id_name(block_type.base_name, atom_name)
            for atom_name in atom_names
        ]
        wildcard_ids = [
            self.get_atom_wildcard_id_name(atom_name) for atom_name in atom_names
        ]
        return unique_ids, wildcard_ids

    def setup_block_type(self, block_type: RefinedResidueType):
        super(AtomTypeDependentTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "atom_types"):
            assert hasattr(block_type, "heavy_atom_inds")
            assert hasattr(block_type, "atom_unique_ids")
            assert hasattr(block_type, "atom_wildcard_ids")
            return

        unique_ids, wildcard_ids = self._create_uniq_and_wildcard_names_for_bt(
            block_type
        )

        atom_types = self.atom_type_index.get_indexer(
            [x.atom_type for x in block_type.atoms]
        )
        heavy_inds = numpy.nonzero(self.np_is_heavyatom[atom_types])[0]

        setattr(block_type, "atom_types", atom_types)
        setattr(block_type, "heavy_atom_inds", heavy_inds)
        setattr(block_type, "atom_unique_ids", unique_ids)
        setattr(block_type, "atom_wildcard_ids", wildcard_ids)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(AtomTypeDependentTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "atom_types"):
            assert hasattr(packed_block_types, "n_heavy_atoms")
            assert hasattr(packed_block_types, "heavy_atom_inds")
            assert hasattr(packed_block_types, "atom_unique_ids")
            assert hasattr(packed_block_types, "atom_wildcard_ids")
            assert hasattr(packed_block_types, "atom_unique_id_index")
            return

        # TO DO: Figure out why this add was necessary
        for bt in packed_block_types.active_block_types:
            self.setup_block_type(bt)

        atom_types = numpy.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms),
            -1,
            dtype=numpy.int32,
        )

        atom_unique_ids = numpy.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms),
            -1,
            dtype=numpy.int32,
        )
        atom_wildcard_ids = numpy.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms),
            -1,
            dtype=numpy.int32,
        )

        atom_unique_id_index = OrderedDict()
        for i, bt in enumerate(packed_block_types.active_block_types):
            for j, atom_name in enumerate(bt.atom_unique_ids):
                if atom_name not in atom_unique_id_index:
                    atom_unique_id_index[atom_name] = len(atom_unique_id_index)
                atom_unique_ids[i, j] = atom_unique_id_index[atom_name]
            for j, atom_name in enumerate(bt.atom_wildcard_ids):
                if atom_name not in atom_unique_id_index:
                    atom_unique_id_index[atom_name] = len(atom_unique_id_index)
                atom_wildcard_ids[i, j] = atom_unique_id_index[atom_name]

        for i, restype in enumerate(packed_block_types.active_block_types):
            atom_types[i, : packed_block_types.n_atoms[i]] = (
                self.atom_type_index.get_indexer([x.atom_type for x in restype.atoms])
            )

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
        atom_unique_ids = torch.tensor(atom_unique_ids, device=self.device)
        atom_wildcard_ids = torch.tensor(atom_wildcard_ids, device=self.device)

        setattr(packed_block_types, "atom_types", atom_types)
        setattr(packed_block_types, "n_heavy_atoms", n_heavy_atoms)
        setattr(packed_block_types, "heavy_atom_inds", heavy_atom_inds_t)
        setattr(packed_block_types, "atom_unique_ids", atom_unique_ids)
        setattr(packed_block_types, "atom_wildcard_ids", atom_wildcard_ids)
        setattr(packed_block_types, "atom_unique_id_index", atom_unique_id_index)

    def setup_poses(self, pose_stack: PoseStack):
        super(AtomTypeDependentTerm, self).setup_poses(pose_stack)
