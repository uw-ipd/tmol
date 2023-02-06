import attr
import numpy
import numba
import torch
import pandas

from tmol.database import ParameterDatabase
from tmol.database.scoring.hbond import HBondDatabase

from tmol.chemical.restypes import RefinedResidueType
from tmol.pose.packed_block_types import PackedBlockTypes

from tmol.score.hbond.params import HBondParamResolver
from tmol.score.common.stack_condense import arg_tile_subset_indices
from tmol.score.chemical_database import AtomTypeParamResolver
from tmol.score.bond_dependent_term import BondDependentTerm

from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray
from tmol.types.torch import Tensor


@numba.jit(nopython=True)
def attached_H_for_don(atom_is_hydrogen, D_idx, bonds, bond_spans):
    donH = numpy.full(atom_is_hydrogen.shape, -1, dtype=numpy.int32)
    heavy_at_don_for_H = numpy.full(atom_is_hydrogen.shape, -1, dtype=numpy.int32)
    which_H_for_hvy = numpy.full(atom_is_hydrogen.shape, -1, dtype=numpy.int32)
    n_attached_H_for_D = numpy.full(D_idx.shape, 0, dtype=numpy.int32)

    n_donH = 0
    for i, d in enumerate(D_idx):
        count_H_for_d = 0
        for b_ind in range(bond_spans[d, 0], bond_spans[d, 1]):
            neighb = bonds[b_ind, 1]
            if atom_is_hydrogen[neighb]:
                donH[n_donH] = neighb
                heavy_at_don_for_H[n_donH] = d
                which_H_for_hvy[n_donH] = count_H_for_d
                count_H_for_d += 1
                n_donH += 1
        n_attached_H_for_D[i] = count_H_for_d
    donH = donH[:n_donH]
    heavy_at_don_for_H = heavy_at_don_for_H[:n_donH]
    which_H_for_hvy = which_H_for_hvy[:n_donH]

    sort_inds = numpy.argsort(donH)
    return (
        donH[sort_inds],
        heavy_at_don_for_H[sort_inds],
        which_H_for_hvy[sort_inds],
        n_attached_H_for_D,
    )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondBlockTypeParams(ValidateAttrs):
    donH_inds: NDArray[numpy.int32][:]
    don_hvy_inds: NDArray[numpy.int32][:]
    acc_inds: NDArray[numpy.int32][:]
    acc_hybridization: NDArray[numpy.int32][:]
    n_donH_for_at: NDArray[numpy.int32][:]

    tile_n_donH: NDArray[numpy.int32][:]
    tile_n_don_hvy: NDArray[numpy.int32][:]
    tile_n_acc: NDArray[numpy.int32][:]
    tile_donH_inds: NDArray[numpy.int32][:, :]  # the tile-ind of a particular donH
    tile_donH_hvy_inds: NDArray[numpy.int32][
        :, :
    ]  # the res-ind of hvy for a particular donH
    tile_don_hvy_inds: NDArray[numpy.int32][
        :, :
    ]  # the tile-ind of a particular donor hvy
    tile_which_donH_of_donH_hvy: NDArray[numpy.int32][
        :, :
    ]  # ind [0..nattachedH) for donH for corr hvy

    tile_acc_inds: NDArray[numpy.int32][:, :]  # the tile-ind of a particular acc
    tile_donorH_type: NDArray[numpy.int32][:, :]
    tile_acceptor_type: NDArray[numpy.int32][:, :]
    tile_acceptor_hybridization: NDArray[numpy.int32][:, :]
    tile_acceptor_n_attached_H: NDArray[numpy.int32][:, :]
    is_hydrogen: NDArray[numpy.int32][:]
    # possible additions? tile_don_hvy_n_attached_H, tile_don_hvy_type


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondPackedBlockTypesParams(ValidateAttrs):
    tile_n_donH: Tensor[torch.int32][:, :]
    tile_n_don_hvy: Tensor[torch.int32][:, :]
    tile_n_acc: Tensor[torch.int32][:, :]
    tile_donH_inds: Tensor[torch.int32][:, :, :]
    tile_donH_hvy_inds: Tensor[torch.int32][:, :, :]
    tile_don_hvy_inds: Tensor[torch.int32][:, :, :]
    tile_which_donH_of_donH_hvy: Tensor[torch.int32][:, :, :]
    tile_acc_inds: Tensor[torch.int32][:, :, :]
    tile_donorH_type: Tensor[torch.int32][:, :, :]
    tile_acceptor_type: Tensor[torch.int32][:, :, :]
    tile_acceptor_hybridization: Tensor[torch.int32][:, :, :]
    tile_acceptor_n_attached_H: Tensor[torch.int32][:, :, :]
    is_hydrogen: Tensor[torch.int32][:, :]


class HBondDependentTerm(BondDependentTerm):
    atom_type_resolver: AtomTypeParamResolver
    hbond_database: HBondDatabase
    hbond_resolver: HBondParamResolver
    device: torch.device
    tile_size = 32

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super(HBondDependentTerm, self).__init__(param_db=param_db, device=device)
        self.atom_type_resolver = AtomTypeParamResolver.from_database(
            param_db.chemical, torch.device("cpu")
        )
        self.hbond_database = param_db.scoring.hbond
        self.hbond_resolver = HBondParamResolver.from_database(
            param_db.chemical, self.hbond_database, device
        )
        self.device = device

    def setup_block_type(self, block_type: RefinedResidueType):
        super(HBondDependentTerm, self).setup_block_type(block_type)

        if hasattr(block_type, "hbbt_params"):
            return

        atom_types = [x.atom_type for x in block_type.atoms]
        atom_type_idx = self.atom_type_resolver.type_idx(atom_types)
        atom_type_params = self.atom_type_resolver.params[atom_type_idx]
        ahnp = atom_type_params.acceptor_hybridization.cpu().numpy()
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

        A_idx = numpy.nonzero(is_acc)[0].astype(dtype=numpy.int32)
        is_hydrogen = (
            atom_type_params.is_hydrogen.cpu().numpy().astype(dtype=numpy.int32)
        )
        acc_hybridization = numpy.zeros((block_type.n_atoms,), dtype=numpy.int32)
        acc_hybridization[: len(A_idx)] = ahnp[A_idx]

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
        D_idx = numpy.nonzero(is_don)[0].astype(dtype=numpy.int32)
        indexed_bonds = block_type.intrares_indexed_bonds
        H_idx, D_for_H, which_H_for_Hs_D, n_H_for_D = attached_H_for_don(
            is_hydrogen,
            D_idx,
            indexed_bonds.bonds.cpu().numpy()[0],
            indexed_bonds.bond_spans.cpu().numpy()[0],
        )
        donH_type = don_type[D_for_H]
        n_donH_for_at = numpy.zeros(block_type.n_atoms, dtype=numpy.int32)
        n_donH_for_at[D_idx] = n_H_for_D
        tile_acceptor_n_attached_H = numpy.full(
            (n_tiles, tile_size), -1, dtype=numpy.int32
        )
        tile_acceptor_n_attached_H[is_tiled_acc] = n_donH_for_at[A_idx]

        tiled_donH_orig_inds, tile_n_donH = arg_tile_subset_indices(
            H_idx, tile_size, block_type.n_atoms
        )
        tiled_don_hvy_orig_inds, tile_n_don_hvy = arg_tile_subset_indices(
            D_idx, tile_size, block_type.n_atoms
        )

        assert tile_n_donH.shape[0] == n_tiles
        assert tile_n_don_hvy.shape[0] == n_tiles
        tiled_donH_orig_inds = tiled_donH_orig_inds.reshape((n_tiles, tile_size))
        is_tiled_donH = tiled_donH_orig_inds != -1
        tiled_don_hvy_orig_inds = tiled_don_hvy_orig_inds.reshape((n_tiles, tile_size))
        is_tiled_don_hvy = tiled_don_hvy_orig_inds != -1
        # print("is_tiled_don_hvy", is_tiled_don_hvy)
        # print("D_idx", D_idx)

        tile_donH_inds = numpy.full((n_tiles, tile_size), -1, dtype=numpy.int32)
        tile_don_hvy_inds = numpy.copy(tile_donH_inds)
        tile_donH_hvy_inds = numpy.copy(tile_donH_inds)
        tile_donorH_type = numpy.copy(tile_donH_inds)
        tile_which_donH_of_donH_hvy = numpy.copy(tile_donH_inds)

        tile_donH_inds[is_tiled_donH] = H_idx
        tile_don_hvy_inds[is_tiled_don_hvy] = D_idx
        tile_donH_hvy_inds[is_tiled_donH] = D_for_H
        tile_donorH_type[is_tiled_donH] = donH_type
        tile_which_donH_of_donH_hvy[is_tiled_donH] = which_H_for_Hs_D

        hbbt_params = HBondBlockTypeParams(
            donH_inds=H_idx,
            don_hvy_inds=D_idx,
            acc_inds=A_idx,
            acc_hybridization=acc_hybridization,
            n_donH_for_at=n_donH_for_at,
            tile_n_donH=tile_n_donH,
            tile_n_don_hvy=tile_n_don_hvy,
            tile_n_acc=tile_n_acc,
            tile_donH_inds=tile_donH_inds,
            tile_donH_hvy_inds=tile_donH_hvy_inds,
            tile_don_hvy_inds=tile_don_hvy_inds,
            tile_which_donH_of_donH_hvy=tile_which_donH_of_donH_hvy,
            tile_acc_inds=tile_acc_inds,
            tile_donorH_type=tile_donorH_type,
            tile_acceptor_type=tile_acceptor_type,
            tile_acceptor_hybridization=tile_acceptor_hybridization,
            tile_acceptor_n_attached_H=tile_acceptor_n_attached_H,
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

        tile_n_donH = numpy.zeros((pbt.n_types, max_n_tiles), dtype=numpy.int32)
        tile_n_don_hvy = numpy.zeros((pbt.n_types, max_n_tiles), dtype=numpy.int32)
        tile_n_acc = numpy.zeros((pbt.n_types, max_n_tiles), dtype=numpy.int32)
        tile_donH_inds = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size), -1, dtype=numpy.int32
        )
        tile_donH_hvy_inds = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size), -1, dtype=numpy.int32
        )
        tile_don_hvy_inds = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size), -1, dtype=numpy.int32
        )
        tile_which_donH_of_donH_hvy = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size), -1, dtype=numpy.int32
        )
        tile_acc_inds = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size), -1, dtype=numpy.int32
        )
        tile_donorH_type = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size), -1, dtype=numpy.int32
        )
        tile_acceptor_type = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size), -1, dtype=numpy.int32
        )
        tile_acceptor_hybridization = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size), -1, dtype=numpy.int32
        )
        tile_acceptor_n_attached_H = numpy.full(
            (pbt.n_types, max_n_tiles, tile_size), -1, dtype=numpy.int32
        )
        is_hydrogen = numpy.full(
            (pbt.n_types, pbt.max_n_atoms), False, dtype=numpy.int32
        )

        for i, block_type in enumerate(packed_block_types.active_block_types):
            i_hb_params = block_type.hbbt_params
            i_n_tiles = i_hb_params.tile_n_donH.shape[0]
            i_n_ats = block_type.n_atoms

            tile_n_donH[i, :i_n_tiles] = i_hb_params.tile_n_donH
            tile_n_don_hvy[i, :i_n_tiles] = i_hb_params.tile_n_don_hvy
            tile_n_acc[i, :i_n_tiles] = i_hb_params.tile_n_acc
            tile_donH_inds[i, :i_n_tiles] = i_hb_params.tile_donH_inds
            tile_donH_hvy_inds[i, :i_n_tiles] = i_hb_params.tile_donH_hvy_inds
            tile_don_hvy_inds[i, :i_n_tiles] = i_hb_params.tile_don_hvy_inds
            tile_which_donH_of_donH_hvy[
                i, :i_n_tiles
            ] = i_hb_params.tile_which_donH_of_donH_hvy
            tile_acc_inds[i, :i_n_tiles] = i_hb_params.tile_acc_inds
            tile_donorH_type[i, :i_n_tiles] = i_hb_params.tile_donorH_type
            tile_acceptor_type[i, :i_n_tiles] = i_hb_params.tile_acceptor_type
            tile_acceptor_hybridization[
                i, :i_n_tiles
            ] = i_hb_params.tile_acceptor_hybridization
            tile_acceptor_n_attached_H[
                i, :i_n_tiles
            ] = i_hb_params.tile_acceptor_n_attached_H
            is_hydrogen[i, :i_n_ats] = i_hb_params.is_hydrogen

        def _tint32(arr):
            return torch.tensor(arr, dtype=torch.int32, device=self.device)

        params = HBondPackedBlockTypesParams(
            tile_n_donH=_tint32(tile_n_donH),
            tile_n_don_hvy=_tint32(tile_n_don_hvy),
            tile_n_acc=_tint32(tile_n_acc),
            tile_donH_inds=_tint32(tile_donH_inds),
            tile_donH_hvy_inds=_tint32(tile_donH_hvy_inds),
            tile_don_hvy_inds=_tint32(tile_don_hvy_inds),
            tile_which_donH_of_donH_hvy=_tint32(tile_which_donH_of_donH_hvy),
            tile_acc_inds=_tint32(tile_acc_inds),
            tile_donorH_type=_tint32(tile_donorH_type),
            tile_acceptor_type=_tint32(tile_acceptor_type),
            tile_acceptor_hybridization=_tint32(tile_acceptor_hybridization),
            tile_acceptor_n_attached_H=_tint32(tile_acceptor_n_attached_H),
            is_hydrogen=_tint32(is_hydrogen),
        )

        setattr(packed_block_types, "hbpbt_params", params)
