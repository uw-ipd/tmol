import attr
import numpy
import torch
import pandas  # ??

from tmol.database import ParameterDatabase
from tmol.database.scoring.hbond import HBondDatabase
from tmol.score.hbond.params import HBondParamResolver
from tmol.database.chemical import ChemicalDatabase
from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.system.restypes import RefinedResidueType
from tmol.system.pose import PackedBlockTypes
from tmol.score.BondDependentTerm import BondDependentTerm


@attr.s(auto_attribs=True)
class HBondDependentTerm(BondDependentTerm):
    atom_resolver: AtomTypeParamResolver
    hbond_database: HBondDatabase
    hbond_resolver: HBondParamResolver
    device: torch.device

    @classmethod
    def from_database(cls, database: ParameterDatabase, device: torch.device):
        atom_resolver = AtomTypeParamResolver.from_database(
            database.chemical, torch.device("cpu")
        )
        hbdb = database.scoring.hbond
        hbond_resolver = HBondParamResolver.from_database(
            database.chemical, hbdb, device
        )
        return cls.from_param_resolvers(
            atom_resolver=atom_resolver,
            hbond_database=HBondDatabase,
            hbond_resolver=hbond_resolver,
            device=device,
        )

    @classmethod
    def from_param_resolvers(
        cls,
        atom_resolver: AtomTypeParamResolver,
        hbond_database: HBondDatabase,
        hbond_resolver: HBondParamResolver,
        device: torch.device,
    ):
        return cls(
            atom_resolver=atom_resolver,
            hbond_database=hbond_database,
            hbond_resolver=hbond_resolver,
            device=device,
        )

    def setup_block_type(self, block_type: RefinedResidueType):
        super(HBondDependentTerm, self).setup_block_type(block_type)

        print("bond spans")
        print(block_type.intrares_indexed_bonds.bond_spans)
        print(
            block_type.intrares_indexed_bonds.bond_spans[:, 1]
            - block_type.intrares_indexed_bonds.bond_spans[:, 0]
        )

        if hasattr(block_type, "is_acceptor"):
            assert hasattr(block_type, "is_donor")
            assert hasattr(block_type, "acceptor_type")
            assert hasattr(block_type, "donor_type")
            assert hasattr(block_type, "acceptor_hybridization")
            assert hasattr(block_type, "is_hydrogen")  # this might collide?
            assert hasattr(block_type, "donor_attached_hydrogens")
            return

        atom_types = [x.atom_type for x in block_type.atoms]
        atom_type_idx = self.atom_resolver.type_idx(atom_types)
        atom_type_params = self.atom_resolver.params[atom_type_idx]
        atom_acceptor_hybridization = atom_type_params.acceptor_hybridization.numpy().astype(
            numpy.int64
        )[
            None, :
        ]

        def map_names(mapper, col_name, type_index):
            is_hbtype = numpy.full(len(atom_types), 0, dtype=numpy.int32)
            hbtype_ind = numpy.full(len(atom_types), 0, dtype=numpy.int32)
            hbtype_names = numpy.full(len(atom_types), None, dtype=object)
            try:
                # if there are no atoms that register as acceptors/donors,
                # pandas will throw a KeyError (annoying!)
                hbtype_df = mapper.loc[atom_types][col_name]
                hbtype_df = hbtype_df.where((pandas.notnull(hbtype_df)), None)
                hbtype_names[:] = numpy.array(hbtypes_df)
            except KeyError:
                pass
            hbtype_ind = type_index.get_indexer(hbtypes)
            is_hbtype = hbtype != -1

            return is_hbtype, hbtype

        is_acc, acc_type = map_names(
            self.hbond_database.acceptor_type_mapper,
            "acc_type",
            self.hbond_resolver.acceptor_type_index,
        )
        is_don, don_type = map_names(
            self.hbond_database.donor_type_mapper,
            "don_type",
            self.hbond_resolver.donor_type_index,
        )

        A_idx = condense_numpy_inds(is_acc[None, :])
        B_idx = numpy.full_like(A_idx, -1)
        B0_idx = numpy.full_like(A_idx, -1)
        atom_is_hydrogen = atom_type_params.is_hydrogen.numpy()[None, :]

        compiled.id_acceptor_bases(
            torch.from_numpy(A_idx),
            torch.from_numpy(B_idx),
            torch.from_numpy(B0_idx),
            torch.from_numpy(atom_acceptor_hybridization),
            torch.from_numpy(atom_is_hydrogen.astype(numpy.ubyte)),
            block_type.intrares_indexed_bonds.bonds,
            block_type.intrares_indexed_bonds.bond_spans,
        )

        base_inds = numpy.full((block_type.n_atoms, 3), -1, dtype=numpy.int32)
        base_inds[A_idx, 0] = A_idx
        base_inds[A_idx, 1] = B_idx
        base_inds[A_idx, 2] = B0_idx

        setattr(block_type, "is_acceptor", is_acc)
        setattr(block_type, "is_donor", is_don)
        setattr(block_type, "acceptor_type", acc_type)
        setattr(block_type, "donor_type", don_type)
        setattr(block_type, "acceptor_hybridization", atom_acceptor_hybridization)
        setattr(block_type, "is_hydrogen", atom_is_hydrogen)
        setattr(block_type, "donor_attached_hydrogens")

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(HBondDependentTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "is_acceptor"):
            assert hasattr(packed_block_types, "is_donor")
            assert hasattr(packed_block_types, "acceptor_type")
            assert hasattr(packed_block_types, "donor_type")
            assert hasattr(packed_block_types, "acceptor_hybridization")
            assert hasattr(packed_block_types, "is_hydrogen")
            assert hasattr(packed_block_types, "attached_hydrogens")
            return

        is_acceptor = numpy.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms),
            0,
            dtype=numpy.int32,
        )
        acceptor_type = numpy.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms),
            -1,
            dtype=numpy.int32,
        )

        is_donor = numpy.full_like(is_acceptor, 0)
        donor_type = numpy.full_like(acceptor_type, -1)

        for i, restype in enumerate(packed_block_types.active_block_types):
            i_slice = slice(packed_block_types.n_atoms[i])
            is_acceptor[i, i_slice] = restype.is_acceptor
            acceptor_type[i, i_slice] = restype.acceptor_type
            is_donor[i, i_slice] = restype.is_donor
            donor_type[i, i_slice] = restype.donor_type
