import numpy
import numba
import toolz
import torch

from tmol.types.array import NDArray
from tmol.types.torch import Tensor

from tmol.system.restypes import RefinedResidueType
from tmol.system.pose import Poses, PackedBlockTypes
from tmol.pack.packer_task import PackerTask

from tmol.pack.rotamer.single_residue_kintree import (
    construct_single_residue_kintree,
    coalesce_single_residue_kintrees,
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
    samplers: List[ChiSampler, ...],
    chem_db: ChemicalDatabase,
):
    construct_single_residue_kintree(restype)
    for sampler in samplers:
        if sampler.defines_rotamrs_for_rt(rt):
            if hasattr(rt, "mc_fingerprints"):
                if sampler.sampler_name() in rt.mc_fingerprints:
                    continue
            else:
                setattr(rt, "mc_fingerprints", {})

            sc_roots = sampler.first_sc_atom_for_rt(rt)
            mc_ats, mc_at_fingerprints = create_mainchain_fingerprint(
                rt, sc_roots, chem_db
            )
            fingerprint = sorted(mc_at_fingerprints)
            rt.mc_fingerprints[sampler.sampler_name] = (
                mc_ats,
                mc_at_fingerprints,
                fingerprint,
            )


def annotate_packed_block_types(pbt: PackedBlockTypes):
    coalesce_single_residue_kintrees(pbt)


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
                nodes[count] = (
                    nodes_orig[n_nodes_offset_for_rot[j] + k]
                    + n_atoms_offset_for_rot[j]
                )
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

    # Unneeded???
    # knowing how many nodes there are for each rotamer lets us get a mapping
    # from node index to rotamer index.
    # rot_for_node = numpy.zeros(n_nodes, dtype=numpy.int32)
    # rot_for_node[first_node_for_rot[:-1]] = 1
    # rot_for_node = numpy.cumsum(rot_for_node)

    # now update the node indices from the generic (1 residue) indices
    # so that they are specific for the atom ids for the rotamers
    # numpy.concatenate((
    # numpy.zeros((1,),dtype=numpy.int64),
    # n_atoms_offset_for_rot[:-1]))

    # atom_offset_for_node = n_atoms_offset_for_rot[rot_for_node]
    # nodes = nodes + atom_offset_for_node

    scanStartsStack = pbt.kintree_scans[block_ind_for_rot]
    genStartsStack = pbt.kintree_gens[block_ind_for_rot]
    # print("genStartsStack")
    # print(genStartsStack.shape)

    atomStartsStack = numpy.swapaxes(genStartsStack[:, :, 0], 0, 1)
    natomsPerGen = atomStartsStack[1:, :] - atomStartsStack[:-1, :]
    natomsPerGen[natomsPerGen < 0] = 0

    # print("natomsPerGen")
    # print(natomsPerGen)

    cumsumAtomStarts = numpy.cumsum(natomsPerGen, axis=1)
    # print("cumsumAtomStarts 1")
    # print(cumsumAtomStarts)
    atomStartsOffsets = numpy.concatenate(
        (
            numpy.zeros(natomsPerGen.shape[0], dtype=numpy.int64).reshape(-1, 1),
            cumsumAtomStarts[:, :-1],
        ),
        axis=1,
    )

    # print("atomStartsOffsets")
    # print(atomStartsOffsets)

    # atomStartsOffsets = exclusive_cumsum(cumsumAtomStarts).reshape(natomsPerGen.shape)

    ngenStack = numpy.swapaxes(pbt.kintree_n_scans_per_gen[block_ind_for_rot], 0, 1)
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

    nodes_orig = pbt.kintree_nodes[block_ind_for_rot].ravel()
    nodes_orig = nodes_orig[nodes_orig >= 0]

    n_nodes_for_rot = pbt.kintree_n_nodes[block_ind_for_rot]
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
    n_atoms_offset_for_rot: NDArray(numpy.int32)[:],
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
def load_from_rotamers_w_offsets_except_first_node(
    arr: NDArray(numpy.int32)[:, :],
    n_atoms_total: int,
    n_atoms_for_rot: NDArray(numpy.int32)[:],
    n_atoms_offset_for_rot: NDArray(numpy.int32)[:],
):
    compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
    count = 1
    for i in range(n_atoms_for_rot.shape[0]):
        for j in range(n_atoms_for_rot[i]):
            compact_arr[count] = arr[i][j] + (
                n_atoms_offset_for_rot[i] if j != 0 else 0
            )
            count += 1
    return compact_arr


def construct_kintree_for_rotamers(
    pbt: PackedBlockTypes,
    rt_block_inds: NDArray(numpy.int32)[:],
    # rt_for_rot: Tensor(torch.int64)[:],
    n_atoms_total: int,
    n_atoms_for_rot: Tensor(torch.int32)[:],
    n_atoms_offset_for_rot: NDArray(numpy.int32)[:],
):
    n_atoms_for_rot = n_atoms_for_rot.cpu().numpy()

    def nab(func, arr):
        return func(
            arr[rt_block_inds], n_atoms_total, n_atoms_for_rot, n_atoms_offset_for_rot
        )

    kt_ids = nab(load_from_rotamers_w_offsets, pbt.kintree_id)
    kt_doftype = nab(load_from_rotamers, pbt.kintree_doftype)
    kt_parent = nab(load_from_rotamers_w_offsets_except_first_node, pbt.kintree_parent)
    kt_frame_x = nab(load_from_rotamers_w_offsets, pbt.kintree_frame_x)
    kt_frame_y = nab(load_from_rotamers_w_offsets, pbt.kintree_frame_y)
    kt_frame_z = nab(load_from_rotamers_w_offsets, pbt.kintree_frame_z)

    return kt_ids, kt_doftype, kt_parent, kt_frame_x, kt_frame_y, kt_frame_z


def build_rotamers(poses: Poses, task: PackerTask):

    all_restypes = {}
    samplers = set([])

    for one_pose_rlts in task.rlts:
        for rlt in one_pose_rlts:
            for sampler in rlt.chi_samplers:
                samplers.add(sampler)
            for rt in rlt.allowed_restypes:
                if id(rt) not in all_restypes:
                    all_restypes[id(rt)] = rt

    for rt_id, rt in all_restypes.items():
        annotate_restype(rt)

    # rebuild the poses, perhaps, if there are residue types in the task
    # that are absent from the poses' PBT
    for rt in poses.packed_block_types.active_residues:
        assert id(rt) in all_restypes

    pose_rts = set([id(rt) for rt in poses.packed_block_types.active_residues])
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
    # print("rt_block_inds")
    # print(rt_block_inds)
    for i, one_pose_rlts in enumerate(task.rlts):
        for j, rlt in enumerate(one_pose_rlts):
            real_rts[i, j, : len(rlt.allowed_restypes)] = 1
    # print("poses device")
    # print(type(poses.device))
    # print(poses.device)
    real_rts = torch.tensor(real_rts, dtype=torch.int32, device=poses.device)
    nz_real_rts = torch.nonzero(real_rts)

    all_chi_samples = [
        sampler.sample_chi_for_poses(poses, task) for sampler in samplers
    ]

    # ok, now we need to figure out how many rotamers each rt is getting.

    # some rts are not real
    # some rts are real but have zero rotamers -- we will have to build these ourselves

    n_rots_for_all_samples = toolz.reduce(
        torch.add, [samples[0] for samples in all_chi_samples]
    )
    # print(n_rots_for_all_samples)

    zero_sample_rts = (
        n_rots_for_all_samples[nz_real_rts[:, 0], nz_real_rts[:, 1], nz_real_rts[:, 2]]
        == 0
    )
    nz_no_rotamer_samples = nz_real_rts[zero_sample_rts, :]

    n_rots_for_all_samples[
        nz_no_rotamer_samples[:, 0],
        nz_no_rotamer_samples[:, 1],
        nz_no_rotamer_samples[:, 2],
    ] = 1

    # print("n_rots_for_all_samples")
    # print(n_rots_for_all_samples)

    n_rots_for_rt = n_rots_for_all_samples[
        nz_real_rts[:, 0], nz_real_rts[:, 1], nz_real_rts[:, 2]
    ]

    n_rots = torch.sum(n_rots_for_rt)
    rt_for_rot = torch.zeros(n_rots, dtype=torch.int64)
    n_rots_for_all_samples_offsets = torch.cumsum(n_rots_for_rt, dim=0)
    rt_for_rot[n_rots_for_all_samples_offsets[:-1]] = 1
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

    ids, doft, par, fx, fy, fz = construct_kintree_for_rotamers(
        pbt, rt_block_inds, n_atoms_total, n_atoms_for_rot, n_atoms_offset_for_rot
    )

    nodes, scans, gens = construct_scans_for_rotamers(
        pbt, block_ind_for_rot, n_atoms_for_rot, n_atoms_offset_for_rot
    )

    # oof

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