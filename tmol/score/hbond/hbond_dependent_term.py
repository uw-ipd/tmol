import attr
import numpy
import torch
import pandas

from tmol.database import ParameterDatabase
from tmol.database.scoring.hbond import HBondDatabase
from tmol.database.chemical import ChemicalDatabase

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes

from tmol.score.hbond.params import HBondParamResolver
from tmol.score.common.stack_condense import condense_numpy_inds
from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.score.bond_dependent_term import BondDependentTerm

from tmol.utility.cpp_extension import load, modulename, relpaths

from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray
from tmol.types.torch import Tensor


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondBlockTypeParams(ValidateAttrs):
    # eventually this might include the list of donors and acceptors
    is_acceptor: NDArray[numpy.bool][:]
    is_donor: NDArray[numpy.bool][:]
    acceptor_type: NDArray[numpy.int32][:]
    acceptor_hybridization: NDArray[numpy.int32][:]
    acceptor_base_inds: NDArray[numpy.int32][:, 3]
    donor_type: NDArray[numpy.int32][:]
    is_hydrogen: NDArray[numpy.bool][:]
    donor_attached_hydrogens: NDArray[numpy.int32][:, :]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondPackedBlockTypesParams(ValidateAttrs):
    # eventually this might include the list of donors and acceptors
    is_acceptor: Tensor[torch.uint8][:, :]
    is_donor: Tensor[torch.uint8][:, :]
    acceptor_type: Tensor[torch.int32][:, :]
    acceptor_hybridization: Tensor[torch.int32][:, :]
    acceptor_base_inds: Tensor[torch.int32][:, :, 3]
    donor_type: Tensor[torch.int32][:, :]
    is_hydrogen: Tensor[torch.uint8][:, :]
    donor_attached_hydrogens: Tensor[torch.int32][:, :, :]


@attr.s(auto_attribs=True)
class HBondDependentTerm(BondDependentTerm):
    atom_type_resolver: AtomTypeParamResolver
    hbond_database: HBondDatabase
    hbond_resolver: HBondParamResolver
    device: torch.device

    @classmethod
    def from_database(cls, database: ParameterDatabase, device: torch.device):
        atom_type_resolver = AtomTypeParamResolver.from_database(
            database.chemical, torch.device("cpu")
        )
        hbdb = database.scoring.hbond
        hbond_resolver = HBondParamResolver.from_database(
            database.chemical, hbdb, device
        )
        return cls.from_param_resolvers(
            atom_type_resolver=atom_type_resolver,
            hbond_database=hbdb,
            hbond_resolver=hbond_resolver,
            device=device,
        )

    @classmethod
    def from_param_resolvers(
        cls,
        atom_type_resolver: AtomTypeParamResolver,
        hbond_database: HBondDatabase,
        hbond_resolver: HBondParamResolver,
        device: torch.device,
    ):
        return cls(
            atom_type_resolver=atom_type_resolver,
            hbond_database=hbond_database,
            hbond_resolver=hbond_resolver,
            device=device,
        )

    def setup_block_type(self, block_type: RefinedResidueType):
        super(HBondDependentTerm, self).setup_block_type(block_type)

        if hasattr(block_type, "hbbt_params"):
            return

        compiled = load(
            modulename("tmol.score.hbond.identification"),
            relpaths("tmol/score/hbond/identification.py", "identification.pybind.cc"),
        )

        atom_types = [x.atom_type for x in block_type.atoms]
        atom_type_idx = self.atom_type_resolver.type_idx(atom_types)
        atom_type_params = self.atom_type_resolver.params[atom_type_idx]
        atom_acceptor_hybridization = atom_type_params.acceptor_hybridization.numpy().astype(
            numpy.int64
        )[
            None, :
        ]

        def map_names(mapper, col_name, type_index):
            # step 1: map atom type names to hbtype names
            # step 2: map hbtype names to hbtype indices
            is_hbtype = numpy.full(len(atom_types), 0, dtype=numpy.int32)
            hbtype_ind = numpy.full(len(atom_types), 0, dtype=numpy.int32)
            hbtype_names = numpy.full(len(atom_types), None, dtype=object)
            try:
                # if there are no atoms that register as acceptors/donors,
                # pandas will throw a KeyError (annoying!)
                hbtype_df = mapper.loc[atom_types][col_name]
                hbtype_df = hbtype_df.where((pandas.notnull(hbtype_df)), None)
                hbtype_names[:] = numpy.array(hbtype_df)
            except KeyError:
                pass
            hbtype_ind = type_index.get_indexer(hbtype_names)
            is_hbtype = hbtype_ind != -1

            return is_hbtype, hbtype_ind

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
            torch.from_numpy(atom_is_hydrogen).to(torch.bool),
            block_type.intrares_indexed_bonds.bonds,
            block_type.intrares_indexed_bonds.bond_spans,
        )

        base_inds = numpy.full((block_type.n_atoms, 3), -1, dtype=numpy.int32)
        base_inds[A_idx, 0] = A_idx
        base_inds[A_idx, 1] = B_idx
        base_inds[A_idx, 2] = B0_idx

        atom_acceptor_hybridization = atom_acceptor_hybridization.astype(
            numpy.int32
        ).squeeze()

        # now lets get the list of attached hydrogen atoms:
        max_n_attached = torch.max(
            block_type.intrares_indexed_bonds.bond_spans[:, :, 1]
            - block_type.intrares_indexed_bonds.bond_spans[:, :, 0]
        )
        D_idx = condense_numpy_inds(is_don[None, :])
        H_idx = numpy.full(D_idx.shape + (max_n_attached,), -1, dtype=numpy.int64)

        compiled.id_donor_attached_hydrogens(
            torch.from_numpy(D_idx),
            torch.from_numpy(H_idx),
            torch.from_numpy(atom_is_hydrogen).to(torch.bool),
            block_type.intrares_indexed_bonds.bonds,
            block_type.intrares_indexed_bonds.bond_spans,
        )

        donor_attached_hydrogens = numpy.full(
            (block_type.n_atoms, max_n_attached), -1, dtype=numpy.int32
        )
        donor_attached_hydrogens[D_idx, :] = H_idx

        atom_is_hydrogen = atom_is_hydrogen.astype(numpy.bool).squeeze()

        hbbt_params = HBondBlockTypeParams(
            is_acceptor=is_acc,
            is_donor=is_don,
            acceptor_type=acc_type.astype(numpy.int32),
            acceptor_hybridization=atom_acceptor_hybridization,
            acceptor_base_inds=base_inds,
            donor_type=don_type.astype(numpy.int32),
            is_hydrogen=atom_is_hydrogen,
            donor_attached_hydrogens=donor_attached_hydrogens,
        )
        setattr(block_type, "hbbt_params", hbbt_params)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(HBondDependentTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "hbpbt_params"):
            return

        is_acceptor = numpy.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms),
            0,
            dtype=numpy.uint8,
        )
        acceptor_type = numpy.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms),
            -1,
            dtype=numpy.int32,
        )

        is_donor = numpy.full_like(is_acceptor, 0)
        donor_type = numpy.full_like(acceptor_type, -1)
        acceptor_hybridization = numpy.full_like(acceptor_type, -1)
        acceptor_base_inds = numpy.full(
            (packed_block_types.n_types, packed_block_types.max_n_atoms, 3),
            -1,
            dtype=numpy.int32,
        )
        is_hydrogen = numpy.full_like(is_acceptor, 0)

        for i, block_type in enumerate(packed_block_types.active_block_types):
            assert hasattr(block_type, "hbbt_params")

        max_n_attached = max(
            bt.hbbt_params.donor_attached_hydrogens.shape[1]
            for bt in packed_block_types.active_block_types
        )

        donor_attached_hydrogens = numpy.full(
            (
                packed_block_types.n_types,
                packed_block_types.max_n_atoms,
                max_n_attached,
            ),
            -1,
            dtype=numpy.int32,
        )

        for i, block_type in enumerate(packed_block_types.active_block_types):
            i_slice = slice(packed_block_types.n_atoms[i])
            assert hasattr(block_type, "hbbt_params")
            i_hb_params = block_type.hbbt_params

            is_acceptor[i, i_slice] = i_hb_params.is_acceptor
            acceptor_type[i, i_slice] = i_hb_params.acceptor_type
            is_donor[i, i_slice] = i_hb_params.is_donor
            donor_type[i, i_slice] = i_hb_params.donor_type
            acceptor_hybridization[i, i_slice] = i_hb_params.acceptor_hybridization
            acceptor_base_inds[i, i_slice] = i_hb_params.acceptor_base_inds
            is_hydrogen[i, i_slice] = i_hb_params.is_hydrogen

            i_attached_slice = slice(i_hb_params.donor_attached_hydrogens.shape[1])
            donor_attached_hydrogens[
                i, i_slice, i_attached_slice
            ] = i_hb_params.donor_attached_hydrogens

        max_n_attached_h = numpy.max(numpy.sum(donor_attached_hydrogens != -1, axis=2))

        donor_attached_hydrogens = donor_attached_hydrogens[:, :, :max_n_attached_h]

        def _tint32(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        def _tbool(arr):
            return torch.tensor(arr, dtype=torch.uint8, device=self.device)

        params = HBondPackedBlockTypesParams(
            is_acceptor=_tbool(is_acceptor),
            is_donor=_tbool(is_donor),
            acceptor_type=_tint32(acceptor_type),
            acceptor_hybridization=_tint32(acceptor_hybridization),
            acceptor_base_inds=_tint32(acceptor_base_inds),
            donor_type=_tint32(donor_type),
            is_hydrogen=_tbool(is_hydrogen),
            donor_attached_hydrogens=_tint32(donor_attached_hydrogens),
        )

        setattr(packed_block_types, "hbpbt_params", params)
