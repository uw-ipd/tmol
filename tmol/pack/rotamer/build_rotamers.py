import numpy
import numba
import toolz
import torch
import attr

from typing import Tuple

from tmol.types.array import NDArray
from tmol.types.attrs import ValidateAttrs
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.utility.tensor.common_operations import exclusive_cumsum1d, stretch
from tmol.database.chemical import ChemicalDatabase
from tmol.kinematics.datatypes import KinTree
from tmol.system.restypes import RefinedResidueType
from tmol.system.pose import Poses, PackedBlockTypes
from tmol.pack.packer_task import PackerTask
from tmol.pack.rotamer.chi_sampler import ChiSampler

from tmol.pack.rotamer.single_residue_kintree import (
    construct_single_residue_kintree,
    coalesce_single_residue_kintrees,
)
from tmol.pack.rotamer.mainchain_fingerprint import (
    annotate_residue_type_with_sampler_fingerprints,
    find_unique_fingerprints,
)


def exc_cumsum_from_inc_cumsum(cumsum):
    return numpy.concatenate((numpy.zeros(1, dtype=numpy.int64), cumsum[:-1]))


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamerSet(ValidateAttrs):
    n_rots_for_pose: Tensor(torch.int64)[:]
    rot_offset_for_pose: Tensor(torch.int64)[:]
    n_rots_for_block: Tensor(torch.int64)[:, :]
    rot_offset_for_block: Tensor(torch.int64)[:, :]
    pose_for_rot: Tensor(torch.int64)[:]
    block_type_ind_for_rot: Tensor(torch.int64)[:]
    block_ind_for_rot: Tensor(torch.int32)[:]
    coords: Tensor(torch.float32)[:, :, :]


# from tmol.system.restype import RefinedResidueType


# step 1: let the dunbrack library annotate the residue types
# step 2: let the dunbrack library annotate the condensed residue types
# step 3: flatten poses
# step 4: use the chi sampler to get the chi samples for all poses
# step 5: count the number of rotamers per pose
# step 5a: including rotamers that the dunbrack sampler does not provide (e.g. gly)
# step 6: allocate a n_poses x max_n_rotamers x max_n_atoms x 3 tensor
# step 7: create (n_poses * max_n_rotamers * max_n_atoms) x 3 view of coord tensor
# step 8: create parent indexing based on start-position offset + residue-type tree data
# step 9: build kintree
# step 10: take starting coordinates from residue roots
# step 10a: take internal dofs from mainchain atoms
# step 10b: take internal dofs for other atoms from rt icoors
# step 11: refold


# residue type annotations:
#  - icoors
#  - parent index (i.e. tree)
#  - list of atoms whose icoors are controlled by dunbrack
#  - bonds that are part of named torsions
#  - mainchain atoms
#  - rottable set for this RT
#
# condensed residue type annotations:
#  - (all data in residue-type annotations)
#
#
# NOTE: sidechain samplers should be listed in the residue-level packer task
# NOTE: all samplers should be given the same starting data and then a mask
#       to indicate which residues they should operate on


@validate_args
def rebuild_poses_if_necessary(
    poses: Poses, task: PackerTask
):  # -> Tuple[Poses, Tuple[ChiSampler, ...]]:
    all_restypes = {}
    samplers = set([])

    for one_pose_rlts in task.rlts:
        for rlt in one_pose_rlts:
            for sampler in rlt.chi_samplers:
                samplers.add(sampler)
            for rt in rlt.allowed_restypes:
                if id(rt) not in all_restypes:
                    all_restypes[id(rt)] = rt

    samplers = tuple(samplers)

    # rebuild the poses, perhaps, if there are residue types in the task
    # that are absent from the poses' PBT

    pose_rts = set([id(rt) for rt in poses.packed_block_types.active_block_types])
    needs_rebuilding = False
    for rt_id in all_restypes:
        if rt_id not in pose_rts:
            needs_rebuilding = True
            break

    if needs_rebuilding:
        # make sure all the pose's residue types are also included
        for one_pose_res in poses.residues:
            for res in one_pose_res:
                rt = res.residue_type
                if id(rt) not in all_restypes:
                    all_restypes[id(rt)] = rt

        pbt = PackedBlockTypes.from_restype_list(
            [rt for rt_id, rt in all_restypes.items()], poses.packed_block_types.device
        )
        block_type_ind = torch.full_like(poses.block_type_ind, -1)
        # this could be more efficient if we mapped orig_block_type to new_block_type
        for i, res in enumerate(poses.residues):
            block_type_ind[i, : len(res)] = torch.tensor(
                pbt.inds_for_res(res), dtype=torch.int32, device=poses.device
            )
        poses = attr.evolve(
            poses, packed_block_types=pbt, block_type_ind=block_type_ind
        )
    else:
        pbt = poses.packed_block_types

    return poses, samplers


def annotate_restype(
    restype: RefinedResidueType,
    samplers: Tuple[ChiSampler, ...],
    chem_db: ChemicalDatabase,
):
    construct_single_residue_kintree(restype)
    annotate_residue_type_with_sampler_fingerprints(restype, samplers, chem_db)


def annotate_packed_block_types(pbt: PackedBlockTypes):
    coalesce_single_residue_kintrees(pbt)
    find_unique_fingerprints(pbt)


def annotate_everything(
    chem_db: ChemicalDatabase, samplers: Tuple[ChiSampler, ...], pbt: PackedBlockTypes
):
    # annotate residue types and packed block types
    # give the samplers a chance first to annotate the
    # residue types, and pbt
    # then we will call our own annotation functions
    for sampler in samplers:
        for rt in pbt.active_block_types:
            sampler.annotate_residue_type(rt)
        sampler.annotate_packed_block_types(pbt)

    for rt in pbt.active_block_types:
        annotate_restype(rt, samplers, chem_db)
    annotate_packed_block_types(pbt)


@numba.jit(nopython=True)
def update_nodes(
    nodes_orig, genStartsStack, n_nodes_offset_for_rot, n_atoms_offset_for_rot
):
    """Merge the 1-residue-kintree nodes data so that all the rotamers can be
    built in a single generational-segmented-scan call. This has the structure
    of load-balanced search operation.
    """

    n_gens = genStartsStack.shape[1]
    n_nodes = nodes_orig.shape[0]
    n_rotamers = n_atoms_offset_for_rot.shape[0]
    nodes = numpy.zeros(n_nodes, dtype=numpy.int32)
    count = 0
    for i in range(n_gens - 1):
        for j in range(n_rotamers):
            for k in range(genStartsStack[j, i, 0], genStartsStack[j, i + 1, 0]):
                nodes[count] = nodes_orig[n_nodes_offset_for_rot[j] + k]
                if nodes[count] != 0:
                    nodes[count] += n_atoms_offset_for_rot[j]
                count += 1
    return nodes


@numba.jit(nopython=True)
def update_scan_starts(
    n_scans, atomStartsOffsets, scanStartsStack, genStartsStack, ngenStack
):
    n_gens = genStartsStack.shape[1]
    n_rotamers = genStartsStack.shape[0]
    scanStarts = numpy.zeros((n_scans,), dtype=numpy.int32)
    count = 0
    for i in range(n_gens - 1):
        for j in range(n_rotamers):
            for k in range(ngenStack[i, j]):
                scanStarts[count] = (
                    atomStartsOffsets[i, j]
                    + scanStartsStack[j, genStartsStack[j, i, 1] + k]
                )
                count += 1
    return scanStarts


@validate_args
def construct_scans_for_rotamers(
    pbt: PackedBlockTypes,
    block_type_ind_for_rot: NDArray(numpy.int32)[:],
    n_atoms_for_rot: Tensor(torch.int32)[:],
    n_atoms_offset_for_rot: NDArray(numpy.int64)[:],
):

    scanStartsStack = pbt.rotamer_kintree.scans[block_type_ind_for_rot]
    genStartsStack = pbt.rotamer_kintree.gens[block_type_ind_for_rot]

    atomStartsStack = numpy.swapaxes(genStartsStack[:, :, 0], 0, 1)
    natomsPerGen = atomStartsStack[1:, :] - atomStartsStack[:-1, :]
    natomsPerGen[natomsPerGen < 0] = 0

    cumsumAtomStarts = numpy.cumsum(natomsPerGen, axis=1)
    atomStartsOffsets = numpy.concatenate(
        (
            numpy.zeros(natomsPerGen.shape[0], dtype=numpy.int64).reshape(-1, 1),
            cumsumAtomStarts[:, :-1],
        ),
        axis=1,
    )

    ngenStack = numpy.swapaxes(
        pbt.rotamer_kintree.n_scans_per_gen[block_type_ind_for_rot], 0, 1
    )
    ngenStack[ngenStack < 0] = 0
    ngenStackCumsum = numpy.cumsum(ngenStack.reshape(-1), axis=0)

    # jitted function that operates on the CPU; need to figure
    # out how to replace this with a GPU-compatible version
    scanStarts = update_scan_starts(
        ngenStackCumsum[-1],
        atomStartsOffsets,
        scanStartsStack,
        genStartsStack,
        ngenStack,
    )

    nodes_orig = pbt.rotamer_kintree.nodes[block_type_ind_for_rot].ravel()
    nodes_orig = nodes_orig[nodes_orig >= 0]

    n_nodes_for_rot = pbt.rotamer_kintree.n_nodes[block_type_ind_for_rot]
    first_node_for_rot = numpy.cumsum(n_nodes_for_rot)
    n_nodes_offset_for_rot = exc_cumsum_from_inc_cumsum(first_node_for_rot)

    nodes = update_nodes(
        nodes_orig, genStartsStack, n_nodes_offset_for_rot, n_atoms_offset_for_rot
    )

    gen_starts = numpy.sum(genStartsStack, axis=0)

    return nodes, scanStarts, gen_starts


@numba.jit(nopython=True)
def load_from_rotamers(
    arr: NDArray(numpy.int32)[:, :],
    n_atoms_total: int,
    n_atoms_for_rot: NDArray(numpy.int32)[:],
):
    compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
    count = 1
    for i in range(n_atoms_for_rot.shape[0]):
        for j in range(n_atoms_for_rot[i]):
            compact_arr[count] = arr[i][j]
            count += 1
    return compact_arr


@numba.jit(nopython=True)
def load_from_rotamers_w_offsets(
    arr: NDArray(numpy.int32)[:, :],
    n_atoms_total: int,
    n_atoms_for_rot: NDArray(numpy.int32)[:],
    n_atoms_offset_for_rot: NDArray(numpy.int32)[:],
):
    compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
    count = 1
    for i in range(n_atoms_for_rot.shape[0]):
        for j in range(n_atoms_for_rot[i]):
            compact_arr[count] = arr[i][j] + n_atoms_offset_for_rot[i]
            count += 1
    return compact_arr


@numba.jit(nopython=True)
def load_rotamer_parents(
    parents: NDArray(numpy.int32)[:, :],
    n_atoms_total: int,
    n_atoms_for_rot: NDArray(numpy.int32)[:],
    n_atoms_offset_for_rot: NDArray(numpy.int32)[:],
):
    compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
    count = 1
    for i in range(n_atoms_for_rot.shape[0]):
        for j in range(n_atoms_for_rot[i]):
            # offset by 1 for the root node of each rotamer
            # because the parent array has -1 for root nodes'
            # parents and we want to make it 0
            compact_arr[count] = parents[i][j] + (
                n_atoms_offset_for_rot[i] if j != 0 else 1
            )
            count += 1
    return compact_arr


@validate_args
def construct_kintree_for_rotamers(
    pbt: PackedBlockTypes,
    rot_block_type_ind: NDArray(numpy.int32)[:],
    n_atoms_total: int,
    n_atoms_for_rot: Tensor(torch.int32)[:],
    block_offset_for_rot: NDArray(numpy.int32)[:],
    device: torch.device,
):
    """Construct a KinTree for a set of rotamers by stringing
    together the kintree data for individual rotamers.
    The "block_ofset_for_rot" array is used to construct
    the "id" tensor in the KinTree, which maps to the atom
    indices; thus it should contain the atom-index offsets
    for the first atom in each rotamer in the coords tensor
    that will be used to construct the kintree_coords tensor.
    """

    n_atoms_for_rot = n_atoms_for_rot.cpu().numpy()

    # append a 1 for the root node and then treat
    # the resulting (inclusive) scan as if it
    # represents offsets
    temp = numpy.concatenate((numpy.ones(1, dtype=numpy.int32), n_atoms_for_rot))
    n_atoms_offset_for_rot = numpy.cumsum(temp)

    def nab(func, arr):
        return func(arr[rot_block_type_ind], n_atoms_total, n_atoms_for_rot)

    def nab2(func, arr, rot_offset):
        return func(arr[rot_block_type_ind], n_atoms_total, n_atoms_for_rot, rot_offset)

    def _t(arr):
        return torch.tensor(arr, dtype=torch.int32, device=device)

    id = _t(
        nab2(load_from_rotamers_w_offsets, pbt.rotamer_kintree.id, block_offset_for_rot)
    )
    id[0] = -1
    doftype = _t(nab(load_from_rotamers, pbt.rotamer_kintree.doftype))
    parent = _t(
        nab2(load_rotamer_parents, pbt.rotamer_kintree.parent, n_atoms_offset_for_rot)
    )
    frame_x = _t(
        nab2(
            load_from_rotamers_w_offsets,
            pbt.rotamer_kintree.frame_x,
            n_atoms_offset_for_rot,
        )
    )
    frame_y = _t(
        nab2(
            load_from_rotamers_w_offsets,
            pbt.rotamer_kintree.frame_y,
            n_atoms_offset_for_rot,
        )
    )
    frame_z = _t(
        nab2(
            load_from_rotamers_w_offsets,
            pbt.rotamer_kintree.frame_z,
            n_atoms_offset_for_rot,
        )
    )

    return KinTree(
        id=id,
        doftype=doftype,
        parent=parent,
        frame_x=frame_x,
        frame_y=frame_y,
        frame_z=frame_z,
    )


def measure_dofs_from_orig_coords(
    coords: Tensor(torch.float32)[:, :, :], kintree: KinTree
):
    from tmol.kinematics.compiled.compiled import inverse_kin

    kintree_coords = coords.view(-1, 3)[kintree.id.to(torch.int64)]
    kintree_coords[0, :] = 0  # reset root

    dofs_orig = inverse_kin(
        kintree_coords,
        kintree.parent,
        kintree.frame_x,
        kintree.frame_y,
        kintree.frame_z,
        kintree.doftype,
    )
    return dofs_orig


def measure_pose_dofs(poses):
    # measure the DOFs for the original residues
    # -- first build kintrees for the original residues,
    # -- then call measure_dofs_from_orig_coords

    pbt = poses.packed_block_types
    pbti = poses.block_type_ind.view(-1)
    orig_res_block_type_ind = pbti[pbti != -1]
    real_poses_blocks = pbti != -1
    nz_real_poses_blocks = torch.nonzero(real_poses_blocks).flatten()
    orig_atom_offset_for_poses_blocks = (
        nz_real_poses_blocks.cpu().numpy().astype(numpy.int32) * pbt.max_n_atoms
    )

    n_atoms_for_orig = pbt.n_atoms[orig_res_block_type_ind.to(torch.int64)]
    n_atoms_offset_for_orig = torch.cumsum(n_atoms_for_orig, dim=0)
    n_atoms_offset_for_orig = n_atoms_offset_for_orig.cpu().numpy()
    n_orig_atoms_total = n_atoms_offset_for_orig[-1]

    orig_kintree = construct_kintree_for_rotamers(
        poses.packed_block_types,
        orig_res_block_type_ind.cpu().numpy(),
        int(n_orig_atoms_total),
        n_atoms_for_orig,
        orig_atom_offset_for_poses_blocks,
        poses.device,
    )

    # orig_dofs returned in kintree order
    return measure_dofs_from_orig_coords(poses.coords, orig_kintree)


def merge_chi_samples(chi_samples):
    # chi_samples
    # 0. n_rots_for_rt
    # 1. rt_for_rotamer
    # 2. chi_defining_atom_for_rotamer
    # 3. chi_for_rotamers

    # everything needs to be on the same device
    for samples in chi_samples:
        for i in range(1, len(samples)):
            assert samples[0].device == samples[i].device
        assert chi_samples[0][0].device == samples[0].device

    device = chi_samples[0][0].device

    rt_nrot_offsets = []
    for samples in chi_samples:
        rt_nrot_offsets.append(exclusive_cumsum1d(samples[0]).to(torch.int64))

    all_rt_for_rotamer_unsorted = torch.cat([samples[1] for samples in chi_samples])
    n_rotamers = all_rt_for_rotamer_unsorted.shape[0]
    max_n_rotamers_per_rt = max(torch.max(samples[0]).item() for samples in chi_samples)

    for i, samples in enumerate(chi_samples):
        rot_counter_for_rt = (
            torch.arange(samples[1].shape[0], dtype=torch.int64, device=device)
            - rt_nrot_offsets[i][samples[1].to(torch.int64)]
        )
        numpy.testing.assert_array_less(
            rot_counter_for_rt.cpu().numpy(), max_n_rotamers_per_rt
        )

    sort_rt_for_rotamer = torch.cat(
        [
            samples[1].to(torch.int64) * len(chi_samples) * max_n_rotamers_per_rt
            + i * max_n_rotamers_per_rt
            + torch.arange(samples[1].shape[0], dtype=torch.int64, device=device)
            - rt_nrot_offsets[i][samples[1].to(torch.int64)]
            for i, samples in enumerate(chi_samples)
        ]
    )
    sampler_for_rotamer_unsorted = torch.cat(
        [
            torch.full((samples[1].shape[0],), i, dtype=torch.int64)
            for i, samples in enumerate(chi_samples)
        ]
    )
    sort_ind_for_rotamer = torch.argsort(sort_rt_for_rotamer)
    sort_rt_for_rotamer_sorted = sort_rt_for_rotamer[sort_ind_for_rotamer]
    uniq_sort_rt_for_rotamer = torch.unique(sort_rt_for_rotamer_sorted)

    assert uniq_sort_rt_for_rotamer.shape[0] == sort_rt_for_rotamer_sorted.shape[0]

    sampler_for_rotamer = sampler_for_rotamer_unsorted[sort_ind_for_rotamer]

    all_rt_for_rotamer = torch.cat([samples[1] for samples in chi_samples])[
        sort_ind_for_rotamer
    ]

    max_n_chi_atoms = max(samples[2].shape[1] for samples in chi_samples)
    all_chi_atoms = torch.full(
        (n_rotamers, max_n_chi_atoms), -1, dtype=torch.int32, device=device
    )
    all_chi = torch.full(
        (n_rotamers, max_n_chi_atoms), -1, dtype=torch.float32, device=device
    )
    offset = 0
    for samples in chi_samples:
        assert samples[2].shape[0] == samples[3].shape[0]
        all_chi_atoms[
            offset : (offset + samples[2].shape[0]), : samples[2].shape[1]
        ] = samples[2]
        all_chi[
            offset : (offset + samples[2].shape[0]), : samples[3].shape[1]
        ] = samples[3]
        offset += samples[2].shape[0]

    all_chi_atoms = all_chi_atoms[sort_ind_for_rotamer]
    all_chi = all_chi[sort_ind_for_rotamer]

    # ok, now we need to figure out how many rotamers each rt is getting.
    n_rots_for_rt = toolz.reduce(torch.add, [samples[0] for samples in chi_samples])

    return (
        n_rots_for_rt,
        sampler_for_rotamer,
        all_rt_for_rotamer,
        all_chi_atoms,
        all_chi,
    )


@validate_args
def create_dof_inds_to_copy_from_orig_to_rotamers(
    poses: Poses,
    task: PackerTask,
    samplers,  # : Tuple[ChiSampler, ...],
    rt_for_rot: Tensor(torch.int64)[:],
    block_type_ind_for_rot: Tensor(torch.int64)[:],
    sampler_for_rotamer: Tensor(torch.int64)[:],
    n_dof_atoms_offset_for_rot: Tensor(torch.int32)[:],
) -> Tuple[Tensor(torch.int64)[:], Tensor(torch.int64)[:]]:

    # we want to copy from the orig_dofs tensor into the
    # rot_dofs tensor for the "mainchain" atoms in the
    # original residues into the appropriate positions
    # for the rotamers thta we are building at those
    # residues. This requires a good deal of reindexing.

    pbt = poses.packed_block_types
    n_rots = n_dof_atoms_offset_for_rot.shape[0]

    sampler_ind_mapping = torch.tensor(
        [
            pbt.mc_fingerprints.sampler_mapping[sampler.sampler_name()]
            if sampler.sampler_name() in pbt.mc_fingerprints.sampler_mapping
            else -1
            for sampler in samplers
        ],
        dtype=torch.int64,
        device=poses.device,
    )

    sampler_ind_for_rot = sampler_ind_mapping[sampler_for_rotamer]

    orig_block_type_ind = (
        poses.block_type_ind[poses.block_type_ind != -1].view(-1).to(torch.int64)
    )

    poses_res_to_real_poses_res = torch.full(
        (poses.block_type_ind.shape[0] * poses.block_type_ind.shape[1],),
        -1,
        dtype=torch.int64,
        device=poses.device,
    )
    poses_res_to_real_poses_res[poses.block_type_ind.view(-1) != -1] = torch.arange(
        orig_block_type_ind.shape[0], dtype=torch.int64, device=poses.device
    )

    # get the residue index for each rotamer
    max_n_blocks = poses.coords.shape[1]
    res_ind_for_rt = torch.tensor(
        [
            i * max_n_blocks + j
            for i, one_pose_rlts in enumerate(task.rlts)
            for j, rlt in enumerate(one_pose_rlts)
            for _ in rlt.allowed_restypes
        ],
        dtype=torch.int64,
        device=poses.device,
    )
    # res_ind_for_rot = res_ind_for_rt[rt_for_rot]
    real_res_ind_for_rot = poses_res_to_real_poses_res[res_ind_for_rt[rt_for_rot]]

    # look up which mainchain fingerprint each
    # original residue should use

    mcfp = pbt.mc_fingerprints

    sampler_ind_for_orig = mcfp.max_sampler[orig_block_type_ind]
    orig_res_mcfp = mcfp.max_fingerprint[orig_block_type_ind]
    orig_res_mcfp_for_rot = orig_res_mcfp[real_res_ind_for_rot]

    # now lets find the kintree-ordered indices of the
    # mainchain atoms for the rotamers that represents
    # the destination for the dofs we're copying
    max_n_mcfp_atoms = mcfp.atom_mapping.shape[3]

    rot_mcfp_at_inds_rto = mcfp.atom_mapping[
        sampler_ind_for_rot, orig_res_mcfp_for_rot, block_type_ind_for_rot, :
    ].view(-1)

    real_rot_mcfp_at_inds_rto = rot_mcfp_at_inds_rto[rot_mcfp_at_inds_rto != -1]

    real_rot_block_type_ind_for_mcfp_ats = stretch(
        block_type_ind_for_rot, max_n_mcfp_atoms
    )[rot_mcfp_at_inds_rto != -1]

    rot_mcfp_at_inds_kto = torch.full_like(rot_mcfp_at_inds_rto, -1)
    rot_mcfp_at_inds_kto[rot_mcfp_at_inds_rto != -1] = torch.tensor(
        pbt.rotamer_kintree.kintree_idx[
            real_rot_block_type_ind_for_mcfp_ats.cpu().numpy(),
            real_rot_mcfp_at_inds_rto.cpu().numpy(),
        ],
        dtype=torch.int64,
        device=pbt.device,
    )

    rot_mcfp_at_inds_kto[rot_mcfp_at_inds_kto != -1] += n_dof_atoms_offset_for_rot[
        torch.div(
            torch.arange(n_rots * max_n_mcfp_atoms, dtype=torch.int64), max_n_mcfp_atoms
        )[rot_mcfp_at_inds_kto != -1]
    ].to(torch.int64)

    # now get the indices in the orig_dofs array for the atoms to copy from.
    # The steps:
    # 1. get the mainchain atom indices for each of the original residues
    #    in residue-type order (rto)
    # 2. sample 1. for each rotamer
    # 3. find the real subset of these atoms
    # 4. note the residue index for each of these real atoms
    # 5. remap these to kintree order (kto)
    # 6. increment the indices with the original-residue dof-index offsets

    # orig_mcfp_at_inds_for_orig_rto:
    # 1. these are the mainchain fingerprint atoms from the original
    #    residues on the pose
    # 2. they are stored in residue-type order (rto)
    # 3. they are indexed by original residue index

    orig_mcfp_at_inds_rto = mcfp.atom_mapping[
        sampler_ind_for_orig, orig_res_mcfp, orig_block_type_ind, :
    ].view(-1)

    real_orig_block_type_ind_for_orig_mcfp_ats = stretch(
        orig_block_type_ind, max_n_mcfp_atoms
    )[orig_mcfp_at_inds_rto != -1]

    orig_dof_atom_offset = exclusive_cumsum1d(pbt.n_atoms[orig_block_type_ind]).to(
        torch.int64
    )

    orig_mcfp_at_inds_kto = torch.full_like(orig_mcfp_at_inds_rto, -1)
    orig_mcfp_at_inds_kto[orig_mcfp_at_inds_rto != -1] = (
        torch.tensor(
            pbt.rotamer_kintree.kintree_idx[
                real_orig_block_type_ind_for_orig_mcfp_ats.cpu().numpy(),
                orig_mcfp_at_inds_rto[orig_mcfp_at_inds_rto != -1].cpu().numpy(),
            ],
            dtype=torch.int64,
            device=pbt.device,
        )
        + orig_dof_atom_offset[
            torch.div(
                torch.arange(
                    orig_block_type_ind.shape[0] * max_n_mcfp_atoms,
                    dtype=torch.int64,
                    device=pbt.device,
                ),
                max_n_mcfp_atoms,
            )
        ][orig_mcfp_at_inds_rto != -1]
    )

    orig_mcfp_at_inds_kto = orig_mcfp_at_inds_kto.view(
        orig_block_type_ind.shape[0], max_n_mcfp_atoms
    )

    orig_mcfp_at_inds_for_rot_kto = orig_mcfp_at_inds_kto[real_res_ind_for_rot, :].view(
        -1
    )

    # pare down the subset to those where the mc atom is present for
    # both the original block type and the alternate block type;
    # take the subset and also increment the indices of all the atoms
    # by one to take into account the virtual root atom at the origin

    # later versions of torch have this logical_and function...
    # both_present = torch.logical_and(
    #     rot_mcfp_at_inds_kto != -1, orig_mcfp_at_inds_for_rot_kto != -1
    # )
    both_present = (rot_mcfp_at_inds_kto != -1) + (
        orig_mcfp_at_inds_for_rot_kto != -1
    ) == 2
    rot_mcfp_at_inds_kto = rot_mcfp_at_inds_kto[both_present] + 1
    orig_mcfp_at_inds_for_rot_kto = orig_mcfp_at_inds_for_rot_kto[both_present] + 1

    return rot_mcfp_at_inds_kto, orig_mcfp_at_inds_for_rot_kto


@validate_args
def copy_dofs_from_orig_to_rotamers(
    poses: Poses,
    task: PackerTask,
    samplers,  # : Tuple[ChiSampler, ...],
    rt_for_rot: Tensor(torch.int64)[:],
    block_type_ind_for_rot: Tensor(torch.int64)[:],
    sampler_for_rotamer: Tensor(torch.int64)[:],
    n_dof_atoms_offset_for_rot: Tensor(torch.int32)[:],
    orig_dofs_kto: Tensor(torch.float32)[:, 9],
    rot_dofs_kto: Tensor(torch.float32)[:, 9],
):

    dst, src = create_dof_inds_to_copy_from_orig_to_rotamers(
        poses,
        task,
        samplers,
        rt_for_rot,
        block_type_ind_for_rot,
        sampler_for_rotamer,
        n_dof_atoms_offset_for_rot,
    )

    rot_dofs_kto[dst, :] = orig_dofs_kto[src, :]


@validate_args
def assign_dofs_from_samples(
    pbt: PackedBlockTypes,
    rt_for_rot: Tensor(torch.int64)[:],
    block_type_ind_for_rot: Tensor(torch.int64)[:],
    chi_atoms: Tensor(torch.int32)[:, :],
    chi: Tensor(torch.float32)[:, :],
    rot_dofs_kto: Tensor(torch.float32)[:, 9],
):
    assert chi_atoms.shape == chi.shape
    assert rt_for_rot.shape[0] == block_type_ind_for_rot.shape[0]
    assert rt_for_rot.shape[0] == chi_atoms.shape[0]

    n_atoms = pbt.n_atoms[block_type_ind_for_rot]
    n_rots = rt_for_rot.shape[0]

    atom_offset_for_rot = exclusive_cumsum1d(n_atoms)

    max_n_chi_atoms = chi_atoms.shape[1]
    real_atoms = chi_atoms.view(-1) != -1

    rot_ind_for_real_atom = torch.div(
        torch.arange(max_n_chi_atoms * n_rots, dtype=torch.int64, device=pbt.device),
        max_n_chi_atoms,
    )[real_atoms]

    block_type_ind_for_rot_atom = (
        block_type_ind_for_rot[rot_ind_for_real_atom].cpu().numpy()
    )

    rot_chi_atoms_kto = torch.tensor(
        pbt.rotamer_kintree.kintree_idx[
            block_type_ind_for_rot_atom, chi_atoms.view(-1)[real_atoms].cpu().numpy()
        ],
        dtype=torch.int64,
        device=pbt.device,
    )
    # increment with the atom offsets for the source rotamer and by
    # one to include the virtual root

    rot_chi_atoms_kto += atom_offset_for_rot[rot_ind_for_real_atom].to(torch.int64) + 1

    # overwrite the "downstream torsion" for the atoms that control
    # each chi
    rot_dofs_kto[rot_chi_atoms_kto, 3] = chi.view(-1)[real_atoms]


def calculate_rotamer_coords(
    pbt: PackedBlockTypes,
    n_rots: int,
    rot_kintree: KinTree,
    nodes: NDArray(numpy.int32)[:],
    scans: NDArray(numpy.int32)[:],
    gens: NDArray(numpy.int32)[:],
    rot_dofs_kto: Tensor(torch.float32)[:, 9],
):
    def _p(t):
        return torch.nn.Parameter(t, requires_grad=False)

    def _t(t):
        return torch.tensor(t, dtype=torch.int32, device=pbt.device)

    def _tcpu(t):
        return torch.tensor(t, dtype=torch.int32, device=torch.device("cpu"))

    kintree_stack = _p(
        torch.stack(
            [
                rot_kintree.id,
                rot_kintree.doftype,
                rot_kintree.parent,
                rot_kintree.frame_x,
                rot_kintree.frame_y,
                rot_kintree.frame_z,
            ],
            dim=1,
        ).to(pbt.device)
    )

    new_coords_kto = torch.ops.tmol.forward_only_kin_op(
        rot_dofs_kto, _p(_t(nodes)), _p(_t(scans)), _p(_tcpu(gens)), kintree_stack
    )

    new_coords_rto = torch.zeros(
        (n_rots * pbt.max_n_atoms, 3), dtype=torch.float32, device=pbt.device
    )

    new_coords_rto[rot_kintree.id.to(torch.int64)] = new_coords_kto
    new_coords_rto = new_coords_rto.view(n_rots, pbt.max_n_atoms, 3)
    return new_coords_rto


def get_rotamer_origin_data(task: PackerTask, rt_for_rot: Tensor(torch.int32)[:]):
    n_poses = len(task.rlts)
    pose_for_rt = torch.tensor(
        [
            i
            for i, one_pose_rlts in enumerate(task.rlts)
            for rlts in one_pose_rlts
            for rlt in rlts.allowed_restypes
        ],
        dtype=torch.int32,
        device=rt_for_rot.device,
    )

    block_ind_for_rt = torch.tensor(
        [
            j
            for one_pose_rlts in task.rlts
            for j, rlts in enumerate(one_pose_rlts)
            for rlt in rlts.allowed_restypes
        ],
        dtype=torch.int32,
        device=rt_for_rot.device,
    )
    max_n_blocks = max(len(one_pose_rlts) for one_pose_rlts in task.rlts)

    rt_for_rot64 = rt_for_rot.to(torch.int64)
    pose_for_rot = pose_for_rt[rt_for_rot64].to(torch.int64)
    n_rots_for_pose = torch.bincount(pose_for_rot, minlength=len(task.rlts))
    rot_offset_for_pose = exclusive_cumsum1d(n_rots_for_pose)
    block_ind_for_rot = block_ind_for_rt[rt_for_rot64]
    block_ind_for_rt_global = max_n_blocks * pose_for_rt + block_ind_for_rt
    block_ind_for_rot_global = block_ind_for_rt_global[rt_for_rot64]
    n_rots_for_block = torch.bincount(
        block_ind_for_rot_global, minlength=n_poses * max_n_blocks
    ).reshape(n_poses, max_n_blocks)
    rot_offset_for_block = exclusive_cumsum1d(n_rots_for_block.flatten()).reshape(
        n_poses, max_n_blocks
    )

    return (
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        pose_for_rot,
        block_ind_for_rot,
    )


def build_rotamers(poses: Poses, task: PackerTask, chem_db: ChemicalDatabase):

    poses, samplers = rebuild_poses_if_necessary(poses, task)
    pbt = poses.packed_block_types
    annotate_everything(chem_db, samplers, pbt)

    n_sys = poses.coords.shape[0]
    max_n_blocks = poses.coords.shape[1]
    # max_n_rts = max(
    #     len(rts.allowed_restypes)
    #     for one_pose_rlts in task.rlts
    #     for rts in one_pose_rlts
    # )
    rt_names = [
        rt.name
        for one_pose_rlts in task.rlts
        for rlt in one_pose_rlts
        for rt in rlt.allowed_restypes
    ]
    rt_block_type_ind = pbt.restype_index.get_indexer(rt_names).astype(numpy.int32)

    chi_samples = [sampler.sample_chi_for_poses(poses, task) for sampler in samplers]
    merged_samples = merge_chi_samples(chi_samples)
    n_rots_for_rt, sampler_for_rotamer, rt_for_rotamer, chi_atoms, chi = merged_samples

    n_rots = chi_atoms.shape[0]
    rt_for_rot = torch.zeros(n_rots, dtype=torch.int64, device=poses.device)
    n_rots_for_rt_cumsum = torch.cumsum(n_rots_for_rt, dim=0)
    rt_for_rot[n_rots_for_rt_cumsum[:-1]] = 1
    rt_for_rot = torch.cumsum(rt_for_rot, dim=0).cpu().numpy()

    block_type_ind_for_rot = rt_block_type_ind[rt_for_rot]
    block_type_ind_for_rot_torch = torch.tensor(
        block_type_ind_for_rot, dtype=torch.int64, device=pbt.device
    )
    n_atoms_for_rot = pbt.n_atoms[block_type_ind_for_rot_torch]
    n_atoms_offset_for_rot = torch.cumsum(n_atoms_for_rot, dim=0)
    n_atoms_offset_for_rot = n_atoms_offset_for_rot.cpu().numpy()
    n_atoms_total = n_atoms_offset_for_rot[-1]
    n_atoms_offset_for_rot = exc_cumsum_from_inc_cumsum(n_atoms_offset_for_rot)

    rot_kintree = construct_kintree_for_rotamers(
        pbt,
        block_type_ind_for_rot,
        int(n_atoms_total),
        torch.tensor(n_atoms_for_rot, dtype=torch.int32),
        numpy.arange(n_rots, dtype=numpy.int32) * pbt.max_n_atoms,
        pbt.device,
    )

    nodes, scans, gens = construct_scans_for_rotamers(
        pbt, block_type_ind_for_rot, n_atoms_for_rot, n_atoms_offset_for_rot
    )

    orig_dofs_kto = measure_pose_dofs(poses)

    n_rotamer_atoms = torch.sum(n_atoms_for_rot).item()

    rot_dofs_kto = torch.zeros(
        (n_rotamer_atoms + 1, 9), dtype=torch.float32, device=pbt.device
    )

    rot_dofs_kto[1:] = torch.tensor(
        pbt.rotamer_kintree.dofs_ideal[block_type_ind_for_rot].reshape((-1, 9))[
            pbt.atom_is_real.cpu().numpy()[block_type_ind_for_rot].reshape(-1) != 0
        ],
        dtype=torch.float32,
        device=pbt.device,
    )

    rt_for_rot_torch = torch.tensor(rt_for_rot, dtype=torch.int64, device=pbt.device)

    copy_dofs_from_orig_to_rotamers(
        poses,
        task,
        samplers,
        rt_for_rot_torch,
        block_type_ind_for_rot_torch,
        sampler_for_rotamer,
        torch.tensor(n_atoms_offset_for_rot, dtype=torch.int32, device=pbt.device),
        orig_dofs_kto,
        rot_dofs_kto,
    )

    assign_dofs_from_samples(
        pbt,
        rt_for_rot_torch,
        block_type_ind_for_rot_torch,
        chi_atoms,
        chi,
        rot_dofs_kto,
    )

    rotamer_coords = calculate_rotamer_coords(
        pbt, n_rots, rot_kintree, nodes, scans, gens, rot_dofs_kto
    )

    (
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        pose_for_rot,
        block_ind_for_rot,
    ) = get_rotamer_origin_data(task, rt_for_rot_torch)

    return (
        poses,
        RotamerSet(
            n_rots_for_pose=n_rots_for_pose,
            rot_offset_for_pose=rot_offset_for_pose,
            n_rots_for_block=n_rots_for_block,
            rot_offset_for_block=rot_offset_for_block,
            pose_for_rot=pose_for_rot,
            block_type_ind_for_rot=block_type_ind_for_rot_torch,
            block_ind_for_rot=block_ind_for_rot,
            coords=rotamer_coords,
        ),
    )
