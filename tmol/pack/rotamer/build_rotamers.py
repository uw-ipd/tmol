import numpy
import numba
import toolz
import torch

from typing import Tuple

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

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


def exclusive_cumsum(cumsum):
    return numpy.concatenate((numpy.zeros(1, dtype=numpy.int64), cumsum[:-1]))


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


@numba.jit(nopython=True)
def update_nodes(
    nodes_orig, genStartsStack, n_nodes_offset_for_rot, n_atoms_offset_for_rot
):
    """Merge the 1-residue-type nodes data so that all the rotamers can be
    built in a single generational-segmented-scan call"""

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
                # print("i", i, "j", j, "k", k, "gss:", genStartsStack[j, i, 1] + k)
                scanStarts[count] = (
                    atomStartsOffsets[i, j]
                    + scanStartsStack[j, genStartsStack[j, i, 1] + k]
                )
                count += 1
    return scanStarts


def construct_scans_for_rotamers(
    pbt: PackedBlockTypes,
    block_ind_for_rot: NDArray(numpy.int32)[:],
    n_atoms_for_rot: Tensor(torch.int32)[:],
    n_atoms_offset_for_rot: NDArray(numpy.int32)[:],
):

    scanStartsStack = pbt.rotamer_kintree.scans[block_ind_for_rot]
    genStartsStack = pbt.rotamer_kintree.gens[block_ind_for_rot]

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
        pbt.rotamer_kintree.n_scans_per_gen[block_ind_for_rot], 0, 1
    )
    ngenStack[ngenStack < 0] = 0
    ngenStackCumsum = numpy.cumsum(ngenStack.reshape(-1), axis=0)

    n_gens = genStartsStack.shape[1]

    # jitted function that operates on the CPU; need to figure
    # out how to replace this with a GPU-compatible version
    scanStarts = update_scan_starts(
        ngenStackCumsum[-1],
        atomStartsOffsets,
        scanStartsStack,
        genStartsStack,
        ngenStack,
    )

    nodes_orig = pbt.rotamer_kintree.nodes[block_ind_for_rot].ravel()
    nodes_orig = nodes_orig[nodes_orig >= 0]

    n_nodes_for_rot = pbt.rotamer_kintree.n_nodes[block_ind_for_rot]
    first_node_for_rot = numpy.cumsum(n_nodes_for_rot)
    n_nodes_offset_for_rot = exclusive_cumsum(first_node_for_rot)

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


# @numba.jit(nopython=True)
# def load_from_rotamers_w_block_offsets(
#     arr: NDArray(numpy.int32)[:, :],
#     n_atoms_total: int,
#     n_atoms_for_rot: NDArray(numpy.int32)[:],
#     n_atoms_offset_for_rot: NDArray(numpy.int32)[:],
# ):
#     compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
#     count = 1
#     for i in range(n_atoms_for_rot.shape[0]):
#         i_offset = max_n_atoms * i + 1
#         for j in range(n_atoms_for_rot[i]):
#             compact_arr[count] = arr[i][j] + i_offset
#             count += 1
#     return compact_arr


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
    rot_block_inds: NDArray(numpy.int32)[:],
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
        return func(arr[rot_block_inds], n_atoms_total, n_atoms_for_rot)

    def nab2(func, arr, rot_offset):
        return func(arr[rot_block_inds], n_atoms_total, n_atoms_for_rot, rot_offset)

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


def merge_chi_samples(chi_samples):

    all_rt_for_rotamer_unsorted = torch.cat([samples[2] for samples in chi_samples])
    sort_rt_for_rotamer = torch.cat(
        [samples[2] * len(chi_samples) + i for i, samples in enumerate(chi_samples)]
    )
    sampler_for_rotamer_unsorted = torch.cat(
        [torch.full(samples[2].shape[0], i, dtype=torch.int64)]
    )
    sort_ind_for_rotamer = torch.argsort(sort_rt_for_rotamer)
    sampler_for_rotamer = sampler_for_rotamer_unsorted[sort_ind_for_rotamer]

    all_rt_for_rotamers = torch.cat([samples[2] for samples in chi_samples])[
        sort_ind_for_rotamer
    ]

    all_chi_atoms = torch.cat([samples[3] for samples in chi_samples])[
        sort_ind_for_rotamer
    ]

    all_chi = torch.cat([samples[4] for samples in chi_samples])[sort_ind_for_rotamer]

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
def copy_dofs_from_orig_to_rotamers(
    poses: Poses,
    task: PackerTask,
    samplers: Tuple[ChiSampler, ...],
    rt_for_rot: Tensor(torch.int32)[:],
    block_ind_for_rot: Tensor(torch.int32)[:],
    builder_ind_for_rot: Tensor(torch.int32)[:],
    n_dof_atoms_offset_for_rot: Tensor(torch.int32)[:],
    orig_dofs_kto: Tensor(torch.float32)[:],
    rot_dofs_kto: Tensor(torch.float32)[:],
):
    # we want to copy from the orig_dofs tensor into the
    # rot_dofs tensor for the "mainchain" atoms in the
    # original residues into the appropriate positions
    # for the rotamers thta we are building at those
    # residues. This requires a good deal of reindexing.

    pbt = poses.packed_block_types

    sampler_ind_mapping = torch.tensor(
        [pbt.mc_sampler_mapping[sampler.sampler_name()] for sampler in samplers],
        dtype=torch.int64,
        device=poses.device,
    )

    sampler_mcfp_ind_for_rotamer = sampler_ind_mapping[sampler_for_rotamer]

    # get the residue index for each rotamer
    max_n_blocks = poses.coords.shape[1]
    res_ind_for_rt = torch.tensor(
        [
            i * max_n_blocks + j
            for i, one_pose_rlts in enumerate(task.rlts)
            for j in range(len(one_pose_rlts))
        ],
        dtype=torch.int64,
        device=poses.device,
    )
    res_ind_for_rot = res_ind_for_rt[rt_for_res]

    # look up which mainchain fingerprint each
    # original residue should use
    orig_block_inds = poses.block_inds[poses.block_inds != -1].view(-1).to(torch.int64)

    builder_ind_for_orig = pbt.max_sampler_for_rt[orig_block_inds]
    orig_res_mcfp = pbt.mc_max_fingerprint[orig_block_inds]

    # now lets find the kintree-ordered indices of the
    # mainchain atoms for the rotamers that represents
    # the destination for the dofs we're copying
    max_n_mcfp_atoms = pbt.mc_atom_mapping.shape[3]
    rot_mcfp_at_inds_rto = pbt.mc_atom_mapping[
        builder_ind_for_rot, sampler_mcfp_ind_for_rot, block_ind_for_rot, :
    ].view(-1)
    real_rot_mcfp_at_inds_rto = rot_mcfp_at_inds_rto[rot_mcfp_at_inds_rto != -1]
    real_rot_block_ind_for_mcfp_ats = block_ind_for_rot.repeat(max_n_mcfp_atoms)[
        rot_mcfp_at_inds_rto != -1
    ]

    rot_mcfp_at_inds_kto = torch.full_like(rot_mcfp_at_inds_rto, -1)
    rot_mcfp_at_inds_kto[rot_mcfp_at_inds_rto != -1] = pbt.rotamer_trie.kintree_idx[
        real_rot_block_ind_for_mcfp_ats, real_rot_mcfp_at_inds_rto
    ]

    rot_mcfp_at_inds_kto[rot_mcfp_at_inds_kto != -1] += n_dof_atoms_offset_for_rot[
        torch.arange(n_rots, dtype=torch.int64).repeat(max_n_mcfp_atoms)[
            rot_mcfp_at_inds_kto != -1
        ]
    ]

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

    orig_mcfp_at_inds_rto = pbt.mc_atom_mapping[
        builder_ind_for_orig, orig_res_mcfp, orig_block_inds:
    ].view(-1)

    # nah; delay this orig_dof_at_inds_for_orig_for_rot_rto = orig_dof_at_inds_for_orig_rto[
    # nah; delay this     orig_res_for_rot, :
    # nah; delay this ]

    real_orig_block_ind_for_orig_mcfp_ats = orig_block_inds.repeat(max_n_mcfp_atoms)[
        orig_mcfp_at_inds_rto != -1
    ]

    orig_dof_atom_offset = exclusive_cumsum1d(pbt.n_atoms[orig_block_inds])

    orig_mcfp_at_inds_kto = torch.full_like(orig_mcfp_at_inds_rto, -1)
    orig_mcfp_at_inds_kto[orig_mcfp_at_inds_rto != -1] = (
        pbt.rotamer_tree.kintree_idx[
            real_orig_block_ind_for_orig_mcfp_ats,
            orig_mcfp_at_inds_rto[orig_mcfp_at_inds_rto != -1],
        ]
        + orig_dof_atom_offset
    )

    poses_res_to_real_poses_res = torch.full(
        poses.block_inds.shape[0] * poses.block_inds.shape[1],
        -1,
        dtype=torch.int64,
        device=poses.device,
    )
    poses_res_to_real_poses_res[poses.block_inds.view(-1) != -1] = torch.arange(
        orig_block_inds.shape[0], dtype=torch.int64
    )
    orig_mcfp_at_inds_for_rot_kto = orig_mcfp_at_inds_kto[
        poses_res_to_real_poses_res[res_ind_for_rt[rt_for_rot]], :
    ]

    # pare down the subset to those where the mc atom is present for
    # both the original block type and the alternate block type
    both_present = torch.logical_and(
        rot_mcfp_at_inds_kto != -1, orig_mcfp_at_inds_for_rot_kto != -1
    )
    rot_mcfp_at_inds_kto = rot_mcfp_at_inds_kto[both_present]
    orig_mcfp_at_inds_for_rot_kto = orig_mcfp_at_inds_for_rot_kto[both_present]

    # the big copy we've all been waiting for!
    rot_dofs_kto[rot_mcfp_at_inds_kto, :] = orig_dofs_kto[
        orig_mcfp_at_inds_for_rot_kto, :
    ]


def build_rotamers(poses: Poses, task: PackerTask, chem_db: ChemicalDatabase):

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
    for rt_id, rt in all_restypes.items():
        annotate_restype(rt, samplers, chem_db)

    # rebuild the poses, perhaps, if there are residue types in the task
    # that are absent from the poses' PBT
    for rt in poses.packed_block_types.active_block_types:
        assert id(rt) in all_restypes

    pose_rts = set([id(rt) for rt in poses.packed_block_types.active_block_types])
    needs_rebuilding = False
    for rt_id in all_restypes:
        if rt_id not in pose_rts:
            needs_rebuilding = True
            break

    if needs_rebuilding:
        pbt = PackedBlockTypes.from_restype_list(list(all_restypes))
        block_inds = torch.full_like(poses.block_inds)
        for i, res in enumerate(poses.residues):
            block_inds[i, : len(res)] = torch.tensor(
                pbt.inds_for_res(res), dtype=torch.int32, device=poses.device
            )
        poses = attr.evolve(poses, packed_block_types=pbt, block_inds=block_inds)
    else:
        pbt = poses.packed_block_types

    annotate_packed_block_types(pbt)

    for sampler in samplers:
        for rt_id, rt in all_restypes.items():
            sampler.annotate_residue_type(rt)
        sampler.annotate_packed_block_types(pbt)

    n_sys = poses.coords.shape[0]
    max_n_blocks = poses.coords.shape[1]
    max_n_rts = max(
        len(rts.allowed_restypes)
        for one_pose_rlts in task.rlts
        for rts in one_pose_rlts
    )
    real_rts = numpy.zeros((n_sys, max_n_blocks, max_n_rts), dtype=numpy.int32)
    rt_names = [
        rt.name
        for one_pose_rlts in task.rlts
        for rlt in one_pose_rlts
        for rt in rlt.allowed_restypes
    ]
    rt_block_inds = pbt.restype_index.get_indexer(rt_names)

    chi_samples = [sampler.sample_chi_for_poses(poses, task) for sampler in samplers]
    merged_samples = merge_chi_samples(chi_samples)
    n_rots_for_rt, sampler_for_rotamer, all_rt_for_rotamer, all_chi_atoms, all_chi = (
        merged_samples
    )

    n_rots = all_chi_atoms.shape[0]
    rt_for_rot = torch.zeros(n_rots, dtype=torch.int64, device=poses.device)
    n_rots_for_all_samples_cumsum = torch.cumsum(n_rots_for_rt, dim=0)
    rots_for_sample_offset = torch.cat(
        torch.zeros(1, dtype=torch.int64, device=poses.device),
        n_rots_for_all_samples_cumsum[:-1],
    )
    rt_for_rot[n_rots_for_all_samples_cumsum[:-1]] = 1
    rt_for_rot = torch.cumsum(rt_for_rot, dim=0).cpu().numpy()

    block_ind_for_rot = rt_block_inds[rt_for_rot]
    block_ind_for_rot_torch = torch.tensor(
        block_ind_for_rot, dtype=torch.int64, device=pbt.device
    )
    n_atoms_for_rot = pbt.n_atoms[block_ind_for_rot_torch]
    n_atoms_offset_for_rot = torch.cumsum(n_atoms_for_rot, dim=0)
    n_atoms_offset_for_rot = n_atoms_offset_for_rot.cpu().numpy()
    n_atoms_total = n_atoms_offset_for_rot[-1]
    n_atoms_offset_for_rot = exclusive_cumsum(n_atoms_offset_for_rot)

    rot_kintree = construct_kintree_for_rotamers(
        pbt, block_ind_for_rot, n_atoms_total, n_atoms_for_rot
    )

    nodes, scans, gens = construct_scans_for_rotamers(
        pbt, block_ind_for_rot, n_atoms_for_rot, n_atoms_offset_for_rot
    )

    # Not clear what the rt_sample_offsets tensor will be needed for
    # rt_sample_offsets = torch.cumsum(
    #     n_rots_for_all_samples.view(-1), dim=0, dtype=torch.int32
    # )
    # n_rotamers = rt_sample_offsets[-1].item()
    # # print("n_rotamers")
    # # print(n_rotamers)
    #
    # rt_sample_offsets[1:] = rt_sample_offsets[:-1]
    # rt_sample_offsets[0] = 0
    # rt_sample_offsets = rt_sample_offsets.view(n_sys, max_n_blocks, max_n_rts)

    # rotamer_coords = torch.zeros((n_rotamers, pbt.max_n_atoms, 3), dtype=torch.float32)

    # measure the DOFs for the original residues

    pbi = poses.block_inds.view(-1)
    orig_res_block_ind = pbi[pbi != -1]
    real_orig_block_inds = orig_res_block_inds != -1
    nz_real_orig_block_inds = torch.nonzero(real_orig_block_inds).flatten()
    orig_atom_offset_for_rot = (
        nz_real_block_inds.cpu().numpy().astype(numpy.int32) * pbt.max_n_atoms
    )

    n_atoms_for_orig = pbt.n_atoms[orig_res_block_ind]
    n_atoms_offset_for_orig = torch.cumsum(n_atoms_for_orig, dim=0)
    n_atoms_offset_for_orig = n_atoms_offset_for_orig.cpu().numpy()
    n_orig_atoms_total = n_atoms_offset_for_orig[-1]

    orig_kintree = construct_kintree_for_rotamers(
        pbt,
        orig_res_block_ind,
        n_orig_atoms_total,
        n_atoms_for_orig,
        orig_atom_offset_for_rot,
        poses.device,
    )

    # orig_dofs returned in kintree order
    orig_dofs_kto = measure_dofs_for_orig_coords(poses.coords, orig_kintree)

    n_rotamer_atoms = torch.sum(n_atoms_for_rot).item()
    rot_dofs_kto = torch.zeros((n_rotamer_atoms + 1, 9), dtype=numpy.float32)
    rot_dofs_kto[1:] = pbt.rotamer_kintree.dofs_ideal[block_ind_for_rot]

    copy_dofs_from_orig_to_rotamers(
        poses,
        task,
        samplers,
        rt_for_rot,
        block_ind_for_rot,
        sampler_for_rot,
        atom_offset_for_rot,
        orig_dofs_kto,
        rot_dofs_kto,
    )

    # TODO
    # assign_dofs_from_samples(
