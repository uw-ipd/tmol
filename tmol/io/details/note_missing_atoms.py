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
    chain_begin: NDArray[numpy.int32][:, :],  # unused for now
    res_types: NDArray[numpy.int32][:, :],
    res_type_variants: NDArray[numpy.int32][:, :],
):

    annotate_packed_block_types_w_canonical_ordering(pdb=packed_block_types)
    canonical_ordering_map = pdb = packed_block_types.canonical_ordering_map
    real_res = res_types != -1
    # TEMP! treat everything as a "mid" termini type
    block_type_inds = canonical_ordering_map[
        res_types[real_res], 1, res_type_variants[real_res]
    ]
    return block_type_inds


def note_missing_atoms(
    packed_block_types: PackedBlockTypes,
    chain_begin: NDArray[numpy.int32][:, :],
    block_types: NDArray[numpy.int32][:, :],
    atom_is_present: NDArray[numpy.int32][:, :, :],
):
    """For each block type present in the input structure,
    note which atoms are present and absent from the input
    set of atoms that are given in canonical atom ordering.
    This will be (in the future) used to build atoms that
    are missing in ideal positions
    """
    pass


def annotate_packed_block_types_w_canonical_ordering(pbt: PackedBlockTypes):
    # TEMP! do everything in numpy for the moment
    # TO DO! Use torch tensors on the device!
    #
    # what we want here is to be able to take four pieces of data:
    #  a. restype by 3-letter code
    #  b. termini assignment
    #  c. disulfide assignment
    #  d. his-d vs his-e assignment
    # and then look up into a tensor exactly which block type we are talking about

    if hasattr(pbt, "canonical_ordering_map"):
        return

    max_n_termini_types = 3  # TEMP! 0=Nterm, 1=mid, 2=Cterm
    max_n_aa_variant_types = 2  # TEMP! CYS=0, CYD=1; HISE=0, HISD=1; all others, 0
    canonical_ordering_map = numpy.full(
        (len(ordered_canonical_aa_types), 3, 2), -1, dtype=numpy.int32
    )

    # TO DO: handle N- and C-termini variants
    var0_inds = pbt.restype_index.get_indexer(ordered_canonical_aa_types)
    canonical_ordering_map[:, 1, 0] = var0_inds
    canonical_ordering_map[
        (cys_co_aa_ind, his_co_aa_ind), 1, 1
    ] = pbt.restype_index.get_indexer(["CYD", "HIS_D"])

    setattr(pbt, "canonical_ordering_map", canonical_ordering_map)
