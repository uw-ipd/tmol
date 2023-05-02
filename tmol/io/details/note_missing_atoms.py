import numpy
import numba
import torch

from tmol.types.array import NDArray
from tmol.io.canonical_ordering import (
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms,
)
from tmo.io.details.disulfide_search import cys_co_aa_ind
from tmol.io.details.his_taut_resolution import (
    HisTautResolution,
    his_co_aa_ind,
    his_ND1_in_co,
    his_NE2_in_co,
    his_HD1_in_co,
    his_HE2_in_co,
    his_HN_in_co,
    his_NH_in_co,
    his_NN_in_co,
    his_CG_in_co,
)
from tmol.pose.packed_block_types import PackedBlockTypes


def assign_block_types(
    packed_block_types: PackedBlockTypes,
    chain_begin: Tensor[torch.int32][:, :],  # unused for now
    res_types: Tensor[torch.int32][:, :],
    res_type_variants: Tensor[torch.int32][:, :],
):

    _annotate_packed_block_types_w_canonical_res_order(packed_block_types)
    canonical_res_ordering_map = packed_block_types.canonical_res_ordering_map
    real_res = res_types != -1
    # TEMP! treat everything as a "mid" termini type
    block_type_inds = canonical_res_ordering_map[
        res_types[real_res], 1, res_type_variants[real_res]
    ]
    return block_type_inds


def take_block_type_atoms_from_canonical(
    packed_block_types: PackedBlockTypes,
    chain_begin: Tensor[torch.int32][:, :],
    block_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    atom_is_present: Tensor[torch.int32][:, :, :],
):
    """Now that we have decided which block type each canonical residue
    is, we want to select only those atoms from the canonically-ordered
    coords and atom_is_present tensors
    In the case of the atom_is_present tensor, we will here forward only
    concern ourselves with the atoms that are missing and not with
    the (perhaps more than one) ways in which an atom can be (tentatively)
    provided; thus we will invert atom_is_present and return the new
    tensor as missing_atoms
    """
    pbt = packed_block_types
    _annotate_packed_block_types_w_canonical_res_order(pbt)
    _annotate_packed_block_types_w_canonical_atom_order(pbt)

    block_types64 = block_types.to(torch.int64)

    n_poses = block_types.shape[0]
    max_n_blocks = block_types.shape[1]
    real_block_types = block_types64 != -1
    real_atoms = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.bool, device=pbt.device
    )
    real_atoms[real_block_types] = pbt.atom_is_real[block_types64[real_block_types]]

    canonical_atom_inds = torch.full(
        (n_poses, max_n_blocks, pbt.max_n_atoms),
        -1,
        dtype=torch.int32,
        device=pbt.device,
    )

    canonical_atom_inds[real_block_types] = pbt.canonical_atom_ind_map[block_types64]
    nz_real_pose_ind, nz_real_block_ind, _ = torch.nonzero(real_atoms, as_tuple=True)

    block_type_coords = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms, 3), dtype=torch.float32, device=device
    )
    missing_atoms = torch.zeros(
        (n_poses, max_n_blocks, pbt.max_n_atoms), dtype=torch.int32, device=pbt.device
    )
    block_type_coords[real_atoms] = coords[
        nz_real_pose_ind, nz_real_block_ind, canonical_atom_inds[real_atoms]
    ]
    missing_atoms[real_atoms] = (
        atom_is_present[
            nz_real_pose_ind, nz_real_block_ind, canonical_atom_inds[real_atoms]
        ]
        == 0
    )

    return block_type_coords, missing_atoms


def _annotate_packed_block_types_w_canonical_res_order(pbt: PackedBlockTypes):
    # TEMP! do everything in numpy for the moment
    # TO DO! Use torch tensors on the device!
    #
    # mapping from canonical restype index to the
    # packed-block-types index for that restype
    #
    # what we want here is to be able to take four pieces of data:
    #  a. restype by 3-letter code
    #  b. termini assignment
    #  c. disulfide assignment
    #  d. his-d vs his-e assignment
    # and then look up into a tensor exactly which block type we are talking about
    #

    if hasattr(pbt, "canonical_res_ordering_map"):
        assert hasattr(pbt, "bt_ind_to_cananonical_ind")
        return

    max_n_termini_types = 3  # TEMP! 0=Nterm, 1=mid, 2=Cterm
    max_n_aa_variant_types = 2  # TEMP! CYS=0, CYD=1; HISE=0, HISD=1; all others, 0

    # forward ordering: from canonical-index + variant --> block-type index
    canonical_ordering_map = numpy.full(
        (len(ordered_canonical_aa_types), 3, 2), -1, dtype=numpy.int32
    )
    bt_ind_to_canonical_ind = numpy.full((pbt.n_types,), -1, dtype=numpy.int32)

    # TO DO: handle N- and C-termini variants
    var0_inds = pbt.restype_index.get_indexer(ordered_canonical_aa_types)
    canonical_ordering_map[:, 1, 0] = var0_inds
    canonical_ordering_map[
        (cys_co_aa_ind, his_co_aa_ind), 1, 1
    ] = pbt.restype_index.get_indexer(["CYD", "HIS_D"])

    # map from the specific block type to the generic canoncial aa index
    for i in range(len(canonical_ordering_map)):
        for j in range(3):
            for k in range(2):
                if canonical_ordering_map[i, j, k] != -1:
                    bt_ind_to_canonical_ind[canonical_ordering_map[i, j, k]] = i

    setattr(pbt, "canonical_res_ordering_map", canonical_ordering_map)
    setattr(pbt, "bt_ind_to_canonical_ind", bt_ind_to_canonical_ind)


def _annotate_packed_block_types_w_canonical_atom_order(pbt: PackedBlockTypes):
    if hasattr(pbt, "canonical_atom_ind_map"):
        return
    canonical_atom_ind = numpy.full(
        (pbt.n_types, pbt.max_n_atoms), -1, dtype=numpy.int32
    )
    canonical_res_ordering_map = pbt.canonical_res_ordering_map.cpu()
    for i, bt in enumerate(pbt.active_block_types):
        canonical_res_ind = pbt.bt_ind_to_canonical_ind[i]
        if canonical_res_ind == -1:
            continue
        canonical_res_name = ordered_canonical_aa_types[canonical_res_ind]
        i_canonical_ordering = ordered_canonical_aa_atoms[canonical_res_name]
        for j, name in enumerate(bt.atoms):
            # probably this would be faster if we used a pandas indexer
            # but this is done only once, so, for now, use the slow form
            canonical_atom_ind[i, j] = i_canonical_ordering.index(name)
    canonical_atom_ind = torch.tensor(
        canonical_atom_ind, dtype=torch.int32, device=pbt.device
    )
    setattr(pbt, "canonical_atom_ind_map", canonical_atom_ind)
