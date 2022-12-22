import attr
import numpy
import numba
import torch
import pandas

from tmol.database import ParameterDatabase
from tmol.database.scoring.hbond import HBondDatabase

# from tmol.database.chemical import ChemicalDatabase

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes

from tmol.score.hbond.params import HBondParamResolver
from tmol.score.common.stack_condense import (
    condense_numpy_inds,
    arg_tile_subset_indices,
)
from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.score.bond_dependent_term import BondDependentTerm

from tmol.utility.cpp_extension import load, modulename, relpaths

from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray
from tmol.types.torch import Tensor


@numba.jit(nopython=True)
def attached_H_for_don(atom_is_hydrogen, D_idx, bonds, bond_spans):
    donH = numpy.full(atom_is_hydrogen.shape, -1, dtype=numpy.int32)
    heavy_at_don_for_H = numpy.full(atom_is_hydrogen.shape, -1, dtype=numpy.int32)
    n_donH = 0
    for d in D_idx:
        for b_ind in range(bond_spans[d, 0], bond_spans[d, 1]):
            neighb = bonds[b_ind, 1]
            if atom_is_hydrogen[neighb]:
                donH[n_donH] = neighb
                heavy_at_don_for_H[n_donH] = d
                n_donH += 1
    donH = donH[:n_donH]
    sort_inds = numpy.argsort(donH)
    return donH[sort_inds], heavy_at_don_for_H[sort_inds]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondBlockTypeParams(ValidateAttrs):
    # is_donorH: NDArray[numpy.bool][:]
    # is_acceptor: NDArray[numpy.bool][:]
    tile_n_donH: NDArray[numpy.int32][:]
    tile_n_acc: NDArray[numpy.int32][:]
    tile_donH_inds: NDArray[numpy.int32][:, :]
    tile_acc_inds: NDArray[numpy.int32][:, :]
    tile_donorH_type: NDArray[numpy.int32][:, :]
    tile_acceptor_type: NDArray[numpy.int32][:, :]
    tile_acceptor_hybridization: NDArray[numpy.int32][:, :]
    is_hydrogen: NDArray[numpy.bool][:]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondPackedBlockTypesParams(ValidateAttrs):
    # is_donorH: Tensor[torch.bool][:, :]
    # is_acceptor: Tensor[torch.bool][:, :]
    tile_n_donH: Tensor[torch.int32][:, :]
    tile_n_acc: Tensor[torch.int32][:, :]
    tile_donH_inds: Tensor[torch.int32][:, :, :]
    tile_acc_inds: Tensor[torch.int32][:, :, :]
    tile_donorH_type: Tensor[torch.int32][:, :, :]
    tile_acceptor_type: Tensor[torch.int32][:, :, :]
    tile_acceptor_hybridization: Tensor[torch.int32][:, :, :]
    is_hydrogen: Tensor[torch.bool][:, :]


@attr.s(auto_attribs=True)
class HBondDependentTerm(BondDependentTerm):
    atom_type_resolver: AtomTypeParamResolver
    hbond_database: HBondDatabase
    hbond_resolver: HBondParamResolver
    device: torch.device
    tile_size = 32

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

        atom_types = [x.atom_type for x in block_type.atoms]
        atom_type_idx = self.atom_type_resolver.type_idx(atom_types)
        atom_type_params = self.atom_type_resolver.params[atom_type_idx]
        ahnp = atom_type_params.acceptor_hybridization.numpy()
        atom_acceptor_hybridization = ahnp.astype(numpy.int32)[None, :]

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

        # A_idx = condense_numpy_inds(is_acc[None, :]).squeeze(0)
        A_idx = numpy.nonzero(is_acc)[0].astype(dtype=numpy.int32)
        is_hydrogen = atom_type_params.is_hydrogen.numpy()

        tile_size = HBondDependentTerm.tile_size
        tiled_acc_orig_inds, tile_n_acc = arg_tile_subset_indices(
            A_idx, tile_size, block_type.n_atoms
        )

        n_tiles = tile_n_acc.shape[0]
        tiled_acc_orig_inds = tiled_acc_orig_inds.reshape(n_tiles, tile_size)
        is_tiled_acc = tiled_acc_orig_inds != -1
        tile_acc_inds = numpy.full((n_tiles, tile_size), -1, dtype=numpy.int32)
        tile_acceptor_type = numpy.copy(tile_acc_inds)
        tile_acceptor_hybridization = numpy.copy(tile_acc_inds)
        tile_acc_inds[is_tiled_acc] = A_idx
        tile_acceptor_type[is_tiled_acc] = acc_type[A_idx]
        tile_acceptor_hybridization[is_tiled_acc] = atom_acceptor_hybridization[
            0, A_idx
        ]

        # now lets get the list of attached hydrogen atoms:
        max_n_attached = torch.max(
            block_type.intrares_indexed_bonds.bond_spans[:, :, 1]
            - block_type.intrares_indexed_bonds.bond_spans[:, :, 0]
        )
        # D_idx = condense_numpy_inds(is_don[None, :])
        D_idx = numpy.nonzero(is_don)[0].astype(dtype=numpy.int32)
        indexed_bonds = block_type.intrares_indexed_bonds
        H_idx, D_for_H = attached_H_for_don(
            is_hydrogen,
            D_idx,
            indexed_bonds.bonds.cpu().numpy()[0],
            indexed_bonds.bond_spans.cpu().numpy()[0],
        )
        donH_type = don_type[D_for_H]

        tiled_donH_orig_inds, tile_n_donH = arg_tile_subset_indices(
            H_idx, tile_size, block_type.n_atoms
        )

        assert n_tiles == tile_n_donH.shape[0]
        tiled_donH_orig_inds = tiled_donH_orig_inds.reshape((n_tiles, tile_size))
        is_tiled_donH = tiled_donH_orig_inds != -1

        tile_donH_inds = numpy.full((n_tiles, tile_size), -1, dtype=numpy.int32)
        tile_donorH_type = numpy.copy(tile_donH_inds)

        tile_donH_inds[is_tiled_donH] = H_idx
        tile_donorH_type[is_tiled_donH] = donH_type

        hbbt_params = HBondBlockTypeParams(
            tile_n_donH=tile_n_donH,
            tile_n_acc=tile_n_acc,
            tile_donH_inds=tile_donH_inds,
            tile_acc_inds=tile_acc_inds,
            tile_donorH_type=tile_donorH_type,
            tile_acceptor_type=tile_acceptor_type,
            tile_acceptor_hybridization=tile_acceptor_hybridization,
            is_hydrogen=is_hydrogen,
        )
        setattr(block_type, "hbbt_params", hbbt_params)

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super(HBondDependentTerm, self).setup_packed_block_types(packed_block_types)

        if hasattr(packed_block_types, "hbpbt_params"):
            return

        pbt = packed_block_types
        tile_size = HBondDependentTerm.tile_size
        max_n_tiles = (pbt.max_n_atoms - 1) // tile_size + 1
        for bt in pbt.active_block_types:
            assert hasattr(bt, "hbbt_params")
            assert bt.hbbt_params.tile_n_donH.shape[0] <= max_n_tiles
            assert bt.hbbt_params.tile_n_acc.shape[0] <= max_n_tiles

        tile_n_donH = numpy.zeros(
            (pbt.n_types, max_n_tiles), dtype=numpy.int32  # consider making this uint8
        )
        tile_n_acc = numpy.zeros(
            (pbt.n_types, max_n_tiles), dtype=numpy.int32  # consider making this uint8
        )
        tile_donH_inds = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size),
            -1,
            dtype=numpy.int32,  # consider making this uint8
        )
        tile_acc_inds = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size),
            -1,
            dtype=numpy.int32,  # consider making this uint8
        )
        tile_donorH_type = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size),
            -1,
            dtype=numpy.int32,  # consider making this uint8
        )
        tile_acceptor_type = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size),
            -1,
            dtype=numpy.int32,  # consider making this uint8
        )
        tile_acceptor_hybridization = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size),
            -1,
            dtype=numpy.int32,  # consider making this uint8
        )
        is_hydrogen = numpy.full(
            (pbt.n_types, pbt.max_n_atoms), False, dtype=numpy.bool
        )

        for i, block_type in enumerate(packed_block_types.active_block_types):
            i_hb_params = block_type.hbbt_params
            i_n_tiles = i_hb_params.tile_n_donH.shape[0]
            i_n_ats = block_type.n_atoms

            tile_n_donH[i, :i_n_tiles] = i_hb_params.tile_n_donH
            tile_n_donH[i, :i_n_tiles] = i_hb_params.tile_n_donH
            tile_n_acc[i, :i_n_tiles] = i_hb_params.tile_n_acc
            tile_donH_inds[i, :i_n_tiles] = i_hb_params.tile_donH_inds
            tile_acc_inds[i, :i_n_tiles] = i_hb_params.tile_acc_inds
            tile_donorH_type[i, :i_n_tiles] = i_hb_params.tile_donorH_type
            tile_acceptor_type[i, :i_n_tiles] = i_hb_params.tile_acceptor_type
            tile_acceptor_hybridization[
                i, :i_n_tiles
            ] = i_hb_params.tile_acceptor_hybridization
            is_hydrogen[i, :i_n_ats] = i_hb_params.is_hydrogen

        def _tint32(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        def _tbool(arr):
            return torch.tensor(arr, dtype=torch.bool, device=self.device)

        params = HBondPackedBlockTypesParams(
            tile_n_donH=_tint32(tile_n_donH),
            tile_n_acc=_tint32(tile_n_acc),
            tile_donH_inds=_tint32(tile_donH_inds),
            tile_acc_inds=_tint32(tile_acc_inds),
            tile_donorH_type=_tint32(tile_donorH_type),
            tile_acceptor_type=_tint32(tile_acceptor_type),
            tile_acceptor_hybridization=_tint32(tile_acceptor_hybridization),
            is_hydrogen=_tbool(is_hydrogen),
        )

        setattr(packed_block_types, "hbpbt_params", params)
