import numpy
import torch

from tmol.io.pdb_parsing import atom_record_dtype
from tmol.pose.pose_stack import PoseStack
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from typing import Optional, Union


@validate_args
def write_pose_stack_pdb(
    pose_stack: PoseStack,
    fname_out: str,
    **kwargs,
):
    """Write a PDB-formatted file to disk given an input PoseStack.
    Optionally, additional arguments may be passed to the inner function
    "atom_records_from_pose_stack," e.g. the chain_ind_for_block and
    chain_labels arguments (which bypass the automatic-chain-detection
    step when deciding which residues are part of the same chain and
    give arbitrary labels to the chains, respectively) through this
    function as kwargs. See documentation for
    tmol.io.write_pose_stack.atom_records_from_pose_stack
    """
    from tmol.io.pdb_parsing import to_pdb

    atom_records = atom_records_from_pose_stack(pose_stack, **kwargs)
    pdbstring = to_pdb(atom_records)
    with open(fname_out, "w") as fid:
        fid.write(pdbstring)


@validate_args
def atom_records_from_pose_stack(
    pose_stack: PoseStack,
    chain_ind_for_block: Optional[Tensor[torch.int64][:, :]] = None,
    chain_labels=None,  # : Optional[Union[NDArray[str][:], NDArray[str][:, :]]] = None,
) -> NDArray[atom_record_dtype][:]:
    """Create a numpy array holding the atom records needed to write a
    PDB file from a PoseStack.

    Now, whereas PoseStack does not have a concept of a "chain," a PDB most
    certainly does. The good news is that "chain" is an emergent concept from
    the set of chemical bonds in the system. This function uses the set of
    chemical bonds and declares any residues that are chemically bonded to be
    part of the same chain (with the exception of disulfide bonds, which often
    span between two chains), and then the Union/Find algorithm from there to
    label each residue with a chain index, with residue 0 always being on chain
    0, and then chain index increasing monotonically with residue index. These
    chain indices are then turned into chain letters starting at 'A.' These
    default chain-handling behaviors can be intercepted by using either or both
    of the two arguments: chain_ind_for_block and chain_labels.

    If chain_ind_for_block is given, then each residue (aka block) will be
    labeled by the chain index indicated instead of relying on the Union/Find
    algorithm on the bond graph. This is especially needed if you have constructed
    a PoseStack using the res_not_connected argument to pose_stack_from_canonical_form
    (or any of the PoseStack-construction functions that call it) to state
    that two adjacent residues belong to the same chain but should not either
    be treated as termini residues or have chemical bonds between them. The
    Union/Find algorithm on the chemical graph will declare such residues to
    be parts of different chains. When constructing such a PoseStack, it is
    recommended to pass "return_chain_ind=True" and then give that tensor back to
    this function when saving that PoseStack in PDB format.
    chain_ind_for_block should be an [n-poses x max-n-residues] tensor.

    If chain_labels is given, then the alphabetical characters for the chains
    will be taken from there instead of in ascending order starting from 'A.'
    For an antibody, e.g., chains are typically labeled 'H' and 'L' instead
    of 'A' and 'B.' chain_labels can either be an [n-poses x max-n-chains]
    numpy array of characters (so that different poses in the PoseStack
    can have different chain labels) or a [max-n-chains] numpy array of
    characters (when each PoseStack has the same chain labels).
    """
    from tmol.io.chain_deduction import chain_inds_for_pose_stack

    if chain_ind_for_block is None:
        chain_ind_for_block = chain_inds_for_pose_stack(pose_stack)
    return atom_records_from_coords(
        pose_stack.packed_block_types,
        chain_ind_for_block,
        pose_stack.block_type_ind64,
        pose_stack.coords,
        pose_stack.block_coord_offset,
        chain_labels,
    )


@validate_args
def atom_records_from_coords(
    pbt: "PackedBlockTypes",
    chain_ind_for_block: Union[Tensor[torch.int32][:, :], NDArray[numpy.int64][:, :]],
    block_types64: Tensor[torch.int64][:, :],
    pose_like_coords: Tensor[torch.float32][:, :, 3],
    block_coord_offset: Tensor[torch.int32][:, :],
    chain_labels=None,  # : Optional[Union[NDArray[str][:], NDArray[str][:, :]]] = None,
) -> NDArray[atom_record_dtype][:]:
    """Create a numpy array holding the atom records needed to write a
    PDB file from the coordinates and block types of a stack of structures,
    laid out in pose-stack form.
    """

    from tmol.io.pdb_parsing import atom_record_dtype
    from tmol.utility.tensor.common_operations import exclusive_cumsum1d

    assert pose_like_coords.shape[0] == chain_ind_for_block.shape[0]
    assert pose_like_coords.shape[0] == block_types64.shape[0]
    assert pose_like_coords.shape[0] == block_coord_offset.shape[0]

    assert block_types64.shape[1] == block_coord_offset.shape[1]
    assert block_types64.shape[1] == chain_ind_for_block.shape[1]

    n_poses = block_coord_offset.shape[0]
    max_n_blocks = block_coord_offset.shape[1]
    max_n_pose_atoms = pose_like_coords.shape[1]

    is_real_block = block_types64 != -1
    real_block_pose, real_block_block = torch.nonzero(is_real_block, as_tuple=True)

    n_block_atoms = torch.zeros(
        (n_poses, max_n_blocks), dtype=torch.int32, device=pbt.device
    )
    n_block_atoms[is_real_block] = pbt.n_atoms[block_types64[is_real_block]]
    block_coord_offset64 = block_coord_offset.to(torch.int64)
    block_for_pose_atom = torch.zeros(
        (n_poses, max_n_pose_atoms), dtype=torch.int64, device=pbt.device
    )
    block_for_pose_atom[
        real_block_pose, block_coord_offset64[real_block_pose, real_block_block]
    ] = 1
    block_for_pose_atom = torch.cumsum(block_for_pose_atom, dim=1) - 1
    pose_for_pose_atom = (
        torch.arange(n_poses, dtype=torch.int64, device=pbt.device)
        .repeat_interleave(max_n_pose_atoms)
        .view(n_poses, max_n_pose_atoms)
    )

    n_pose_atoms = torch.sum(n_block_atoms, dim=1)
    n_atoms_total = torch.sum(n_pose_atoms)

    # so we can index just the atoms out of coords tensor, e.g.
    # we need to know which atoms are real and which are padding
    atom_is_real = torch.tile(
        torch.arange(max_n_pose_atoms, dtype=torch.int64, device=pbt.device),
        (n_poses, 1),
    ) < n_pose_atoms.unsqueeze(dim=1)

    pose_for_real_atom = torch.arange(
        n_poses,
        dtype=torch.int64,
        device=pbt.device,
    ).repeat_interleave(n_pose_atoms, dim=0)
    block_for_atom = torch.zeros(
        (n_poses, max_n_pose_atoms), dtype=torch.int64, device=pbt.device
    )
    block_for_atom[
        real_block_pose, block_coord_offset64[real_block_pose, real_block_block]
    ] = 1
    block_for_atom = torch.cumsum(block_for_atom, dim=1) - 1
    block_for_real_atom = block_for_atom[atom_is_real]

    block_local_atom_index_for_pose_atom = (
        torch.arange(max_n_pose_atoms, dtype=torch.int32, device=pbt.device).repeat(
            (n_poses, 1)
        )
        - block_coord_offset[pose_for_pose_atom, block_for_pose_atom]
    )
    block_local_atom_index_for_real_atom = block_local_atom_index_for_pose_atom[
        atom_is_real
    ]

    pose_atom_offsets = exclusive_cumsum1d(n_pose_atoms)

    # ok, let's move everything to the cpu/numpy from here forward
    # chain_begin = chain_begin.cpu().numpy()
    block_types64 = block_types64.cpu().numpy()
    pose_like_coords = pose_like_coords.cpu().detach().numpy()
    block_coord_offset = block_coord_offset.cpu().numpy()
    is_real_block = is_real_block.cpu().numpy()
    n_block_atoms = n_block_atoms.cpu().numpy()
    n_pose_atoms = n_pose_atoms.cpu().numpy()
    pose_for_real_atom = pose_for_real_atom.cpu().numpy()
    atom_is_real = atom_is_real.cpu().numpy()
    block_for_atom = block_for_atom.cpu().numpy()
    block_for_real_atom = block_for_real_atom.cpu().numpy()
    block_local_atom_index_for_real_atom = (
        block_local_atom_index_for_real_atom.cpu().numpy()
    )
    pose_atom_offsets = pose_atom_offsets.cpu().numpy()

    chain_ind_for_real_atom = chain_ind_for_block[
        pose_for_real_atom, block_for_real_atom
    ]

    results = numpy.empty(n_atoms_total, dtype=atom_record_dtype)
    results["record_name"] = numpy.full((n_atoms_total,), "ATOM  ", dtype=str)
    results["modeli"] = pose_for_real_atom
    results["chaini"] = chain_ind_for_real_atom
    results["resi"] = block_for_atom[atom_is_real] + 1
    results["atomi"] = (
        numpy.arange(n_atoms_total, dtype=int)
        + 1
        - pose_atom_offsets[pose_for_real_atom]
    )
    results["model"] = pose_for_real_atom + 1

    if chain_labels is None:
        chain_labels = numpy.array([x for x in "ABCDEFGHIJKLKMNOPQRSTUVWXY"])

    if len(chain_labels.shape) == 1:
        results["chain"] = chain_labels[chain_ind_for_real_atom]
    elif len(chain_labels.shape) == 2:
        results["chain"] = chain_labels[pose_for_real_atom, chain_ind_for_real_atom]

    # create lookup for atom names
    bt_names = numpy.array([bt.name[:3] for bt in pbt.active_block_types])
    bt_atom_names = numpy.empty((pbt.n_types, pbt.max_n_atoms), dtype=object)
    for i, bt in enumerate(pbt.active_block_types):
        for j, at in enumerate(bt.atoms):
            bt_atom_names[i, j] = at.name

    bt_for_real_atom = block_types64[pose_for_real_atom, block_for_real_atom]
    results["resn"] = bt_names[bt_for_real_atom]
    results["atomn"] = bt_atom_names[
        bt_for_real_atom, block_local_atom_index_for_real_atom
    ]
    real_atom_coords = pose_like_coords[atom_is_real]
    results["x"] = real_atom_coords[:, 0]
    results["y"] = real_atom_coords[:, 1]
    results["z"] = real_atom_coords[:, 2]
    results["insert"] = " "
    results["occupancy"] = 1
    results["b"] = 0

    return results
