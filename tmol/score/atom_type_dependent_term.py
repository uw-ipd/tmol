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

    # TODO: should these be declared/inited here?
    # atom_name_index # = attr.ib()
    # atom_unique_id_index # = attr.ib()

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        atom_type_resolver = AtomTypeParamResolver.from_database(
            param_db.chemical, device=device
        )
        super(AtomTypeDependentTerm, self).__init__(param_db=param_db, device=device)
        self.atom_type_resolver = atom_type_resolver
        self.atom_type_index = atom_type_resolver.index
        self.np_is_hydrogen = atom_type_resolver.params.is_hydrogen.cpu().numpy()
        self.np_is_heavyatom = numpy.logical_not(self.np_is_hydrogen)

        self.atom_name_index = OrderedDict()
        self.atom_unique_id_index = OrderedDict()
        self.device = device

    def get_atom_unique_id_name(self, block_name, atom_name):
        return "UNIQUE_ID:" + block_name + ":" + atom_name

    def get_atom_wildcard_id_name(self, block_name, atom_name):
        return "WILDCARD_ID:" + atom_name

    def add_atom_index(self, block_type):
        atom_names = [x.name for x in block_type.atoms]

        unique_id_indices = []
        name_indices = []
        for atom_name in atom_names:
            unique_id = self.get_atom_unique_id_name(block_type.name3, atom_name)
            self.atom_unique_id_index[unique_id] = len(self.atom_unique_id_index)
            unique_id_indices.append(self.atom_unique_id_index[unique_id])

            wildcard_id = self.get_atom_wildcard_id_name(block_type.name3, atom_name)
            if wildcard_id not in self.atom_unique_id_index:
                self.atom_unique_id_index[wildcard_id] = len(self.atom_unique_id_index)
            name_indices.append(self.atom_unique_id_index[wildcard_id])

        bt_atom_name_index = numpy.full((len(atom_names)), -1, dtype=numpy.int32)
        bt_atom_unique_id_index = numpy.full((len(atom_names)), -1, dtype=numpy.int32)

        for index, unique_id_index in enumerate(unique_id_indices):
            bt_atom_unique_id_index[index] = unique_id_index

        for index, name_index in enumerate(name_indices):
            bt_atom_name_index[index] = name_index

        setattr(block_type, "atom_unique_ids", bt_atom_unique_id_index)
        setattr(block_type, "atom_wildcard_ids", bt_atom_name_index)

    def setup_block_type(self, block_type: RefinedResidueType):
        super(AtomTypeDependentTerm, self).setup_block_type(block_type)
        if hasattr(block_type, "atom_types"):
            assert hasattr(block_type, "heavy_atom_inds")
            return
        # TODO: update the block attr checks here and elsewhere
        self.add_atom_index(block_type)

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

        for i, block_type in enumerate(packed_block_types.active_block_types):
            atom_unique_ids[
                i, : packed_block_types.n_atoms[i]
            ] = block_type.atom_unique_ids
            atom_wildcard_ids[
                i, : packed_block_types.n_atoms[i]
            ] = block_type.atom_wildcard_ids

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
        atom_unique_ids = torch.tensor(atom_unique_ids, device=self.device)

        setattr(packed_block_types, "atom_types", atom_types)
        setattr(packed_block_types, "n_heavy_atoms", n_heavy_atoms)
        setattr(packed_block_types, "heavy_atom_inds", heavy_atom_inds_t)
        setattr(packed_block_types, "atom_unique_ids", atom_unique_ids)

    def setup_poses(self, pose_stack: PoseStack):
        super(AtomTypeDependentTerm, self).setup_poses(pose_stack)
