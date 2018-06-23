from typing import Optional

import attr

import torch
import numpy
import numba
from numba import cuda

from .datatypes import KinTree
from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

# Assign rather than import to respect dynamic numba cuda layout
DeviceNDArray = numba.cuda.devicearray.DeviceNDArray


@attr.s(auto_attribs=True, frozen=True)
class GPUKinTreeReordering(ValidateAttrs):
    """Path plans for parallel kinematic operations.

    The GPUKinTreeReordering class partitions a KinTree into a set of paths so that
    scan can be run on each path 1) from root to leaf for forward kinematics, and
    2) from leaf to root for f1f2 derivative summation. To accomplish this, the
    GPUKinTreeReordering class reorders the atoms from the original KinTree order ("ko")
    where atoms are known by their kintree-index ("ki") into 1) their refold order
    ("ro") where atoms are known by their refold index ("ri") and 2) their
    deriv-sum order ("dso") where atoms are known by their deriv-sum index.

    The GPUKinTreeReordering divides the tree into a set of paths. Along
    each path is a continuous chain of atoms that either 1) require their
    coordinate frames computed as a cumulative product of homogeneous
    transforms for the coordinate update algorithm, or 2) require the
    cumulative sum of their f1f2 vectors for the derivative calculation
    algorithm. In both cases, these paths can be processed efficiently
    on the GPU using an algorithm called "scan" and batches of these paths
    can be processed at once in a variant called "segmented scan."

    Each path in the tree is labeled with a depth: a path with depth
    i may depend on the values computed for atoms with depths 0..i-1.
    All of the paths of the same depth can be processed in a single
    kernel execution with segmented scan.

    In order to divide the tree into these paths, this class constructs
    two reorderings of the atoms: a refold ordering (ro) using refold
    indices (ri) and a derivsum ordering (dso) using derivsum indices (di).
    The original ordering from the kintree is the kintree ordering (ko)
    using kintree indices (ki). The indexing in this code is HAIRY, and
    so all arrays are labled with the indexing they are using. There are
    two sets of interconversion arrays for going from kintree indexing
    to 1) refold indexing, and to 2) derivsum indexing: ki2ri & ri2ki and
    ki2dsi & dsi2ki.

    The algorithm for dividing the tree into paths minimizes the number
    of depths in the tree. For any node in the tree, one of its children
    will be in the same path with it, and all other children will be
    roots of their own subtrees. The minimum-depth division of paths
    is computed by lableling the "branching factor" of each atom. The
    branching factor of an atom is 0 if it has no children; if it does
    have children, it is the largest of the branching factors of what
    the atom designates as its on-path child and one-greater than the
    branching factor of any of its other children. Each node then may
    select the child with the largest branching factor as its on-path
    child to minimize its branching factor. This division produces
    a minimum depth tree of paths.

    The same set of paths is used for both the refold algorithm and
    the derivative summation; the refold algorithm starts at path
    roots and multiplies homogeneous transforms towards the leaves.
    The derivative summation algorithm starts at the leaves and sums
    upwards towards the roots.
    """

    natoms: int

    # Operation path definitions
    subpath_child_ko: Optional[numpy.ndarray]
    non_path_children_ko: Optional[numpy.ndarray]

    # Kinematic refold (forward) data arrays
    ki2ri: Optional[DeviceNDArray]
    ri2ki: Optional[DeviceNDArray]
    non_subpath_parent_ro: Optional[DeviceNDArray]
    is_root_ro: Optional[DeviceNDArray]
    refold_atom_ranges: Optional[DeviceNDArray]

    # Derivative summation (backward) data arrays
    ki2dsi: Optional[DeviceNDArray]
    dsi2ki: Optional[DeviceNDArray]
    is_leaf_dso: Optional[DeviceNDArray]
    non_path_children_dso: Optional[DeviceNDArray]
    derivsum_atom_ranges: Optional[DeviceNDArray]

    # Alex: how do I get validate args for a class's construction method?
    @classmethod
    @validate_args
    def from_kintree(cls, kintree: KinTree, device: torch.device):
        """Setup for operations over KinTree on given device."""
        if device.type != "cuda":
            # TODO: Enable
            # raise ValueError(
            #     f"GPUKinTreeReordering not supported for non-cuda devices."
            #     f" device: {device}"
            # )

            return cls(
                natoms=0,
                subpath_child_ko=None,
                non_path_children_ko=None,
                dsi2ki=None,
                non_subpath_parent_ro=None,
                is_root_ro=None,
                ki2ri=None,
                ri2ki=None,
                refold_atom_ranges=None,
                ki2dsi=None,
                is_leaf_dso=None,
                non_path_children_dso=None,
                derivsum_atom_ranges=None,
            )
        elif device.index is not None and device.index != numba.cuda.get_current_device(
        ).id:
            raise ValueError(
                f"GPUKinTreeReordering target device is not current numba context."
                f" device: {device} current: {numba.cuda.get_current_device()}"
            )

        natoms = kintree.id.shape[0]
        parent_ko = numpy.array(kintree.parent, dtype="i4")

        ### Determine path structure

        # Calculate branch factor for each node
        branching_factor_ko = numpy.full((natoms), -1, dtype="int32")
        subpath_child_ko = numpy.full((natoms), -1, dtype="int32")

        compute_branching_factor(
            natoms=natoms,
            parent=parent_ko,
            out_branching_factor=branching_factor_ko,
            out_branchiest_child=subpath_child_ko,
        )

        # Assign path children for each node

        n_nonpath_children_ko = numpy.full((natoms), 0, dtype="int32")
        is_subpath_root_ko = numpy.full((natoms), False, dtype="bool")
        is_subpath_leaf_ko = numpy.full((natoms), False, dtype="bool")

        mark_path_children_and_count_nonpath_children(
            natoms=natoms,
            parent_ko=parent_ko,
            subpath_child_ko=subpath_child_ko,
            out_n_nonpath_children_ko=n_nonpath_children_ko,
            out_is_subpath_root_ko=is_subpath_root_ko,
            out_is_subpath_leaf_ko=is_subpath_leaf_ko,
        )

        # Get list of non_path children for subpath root
        non_path_children_ko = numpy.full(
            (natoms, max(n_nonpath_children_ko)),
            -1,
            dtype="int32",
        )
        list_nonpath_children(
            natoms=natoms,
            is_subpath_root_ko=is_subpath_root_ko,
            parent_ko=parent_ko,
            out_non_path_children_ko=non_path_children_ko,
        )

        # Mark path depths and path lengths

        subpath_length_ko = numpy.zeros((natoms), dtype="int32")
        derivsum_path_depth_ko = numpy.full((natoms), -1, dtype="int32")
        refold_atom_depth_ko = numpy.zeros((natoms), dtype="int32")

        find_derivsum_path_depths(
            natoms=natoms,
            subpath_child_ko=subpath_child_ko,
            non_path_children_ko=non_path_children_ko,
            is_subpath_root_ko=is_subpath_root_ko,
            out_derivsum_path_depth_ko=derivsum_path_depth_ko,
            out_subpath_length_ko=subpath_length_ko,
        )

        find_refold_path_depths(
            natoms=natoms,
            parent_ko=parent_ko,
            is_subpath_root_ko=is_subpath_root_ko,
            out_refold_atom_depth_ko=refold_atom_depth_ko,
        )

        ### Map from paths and to forward kinematic refold indices
        refold_ordering = RefoldOrdering.determine(
            natoms=natoms,
            refold_atom_depth_ko=refold_atom_depth_ko,
            is_subpath_root_ko=is_subpath_root_ko,
            subpath_length_ko=subpath_length_ko,
            subpath_child_ko=subpath_child_ko,
            parent_ko=parent_ko,
        )

        derivsum_ordering = DerivsumOrdering.determine(
            natoms=natoms,
            derivsum_path_depth_ko=derivsum_path_depth_ko,
            subpath_length_ko=subpath_length_ko,
            is_subpath_leaf_ko=is_subpath_leaf_ko,
            is_subpath_root_ko=is_subpath_root_ko,
            parent_ko=parent_ko,
            non_path_children_ko=non_path_children_ko,
        )

        return cls(
            natoms=natoms,
            subpath_child_ko=subpath_child_ko,
            non_path_children_ko=non_path_children_ko,

            # Kinematic refold (forward) data arrays
            ki2ri=cuda.to_device(refold_ordering.ki2ri),
            ri2ki=cuda.to_device(refold_ordering.ri2ki),
            non_subpath_parent_ro=cuda.to_device(
                refold_ordering.non_subpath_parent
            ),
            is_root_ro=cuda.to_device(refold_ordering.is_subpath_root),
            refold_atom_ranges=cuda.to_device(
                refold_ordering.atom_range_for_depth
            ),

            # Derivative summation (backward) data arrays
            ki2dsi=cuda.to_device(derivsum_ordering.ki2dsi),
            dsi2ki=cuda.to_device(derivsum_ordering.dsi2ki),
            is_leaf_dso=cuda.to_device(derivsum_ordering.is_leaf),
            non_path_children_dso=cuda.to_device(
                derivsum_ordering.non_path_children
            ),
            derivsum_atom_ranges=cuda.to_device(
                derivsum_ordering.atom_range_for_depth
            ),
        )

    def active(self):
        return self.natoms != 0


@numba.jit(nopython=True)
def compute_branching_factor(
        natoms, parent, out_branching_factor, out_branchiest_child
):
    for ii in range(natoms - 1, -1, -1):
        ii_bf = out_branching_factor[ii]
        if ii_bf == -1:
            ii_bf = 0
            out_branching_factor[ii] = ii_bf
        ii_parent = parent[ii]
        if ii == ii_parent:
            continue
        parent_bf = out_branching_factor[ii_parent]
        if parent_bf == -1:
            out_branching_factor[ii_parent] = ii_bf
            out_branchiest_child[ii_parent] = ii
        elif ii_bf >= parent_bf:
            out_branching_factor[ii_parent] = max(ii_bf, parent_bf + 1)
            out_branchiest_child[ii_parent] = ii


@numba.jit(nopython=True)
def find_refold_path_depths(
        natoms,
        parent_ko,
        is_subpath_root_ko,
        out_refold_atom_depth_ko,
):
    for ii in range(natoms):
        ii_parent = parent_ko[ii]
        ii_depth = out_refold_atom_depth_ko[ii_parent]
        if is_subpath_root_ko[ii] and ii_parent != ii:
            ii_depth += 1
        out_refold_atom_depth_ko[ii] = ii_depth


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RefoldOrdering:

    ri2ki: numpy.ndarray
    ki2ri: numpy.ndarray
    is_subpath_root: numpy.ndarray
    non_subpath_parent: numpy.ndarray
    atom_range_for_depth: numpy.ndarray

    @classmethod
    def determine(
            cls,
            natoms,
            refold_atom_depth_ko,
            is_subpath_root_ko,
            subpath_length_ko,
            subpath_child_ko,
            parent_ko,
    ):
        ndepths = max(refold_atom_depth_ko) + 1
        rfo = refold_ordering = cls(
            ri2ki=numpy.full((natoms), -1, dtype="int32"),
            ki2ri=numpy.full((natoms), -1, dtype="int32"),
            atom_range_for_depth=numpy.full(
                (ndepths, 2),
                -1,
                dtype="int32",
            ),
            non_subpath_parent=numpy.full((natoms), -1, dtype="int32"),
            is_subpath_root=numpy.full((natoms), True, dtype="bool"),
        )

        # sum the path lengths at each depth
        depth_offsets = numpy.zeros((ndepths), dtype="int32")
        numpy.add.at(
            depth_offsets, refold_atom_depth_ko[is_subpath_root_ko],
            subpath_length_ko[is_subpath_root_ko]
        )
        depth_offsets[1:] = numpy.cumsum(depth_offsets)[:-1]
        depth_offsets[0] = 0

        for i in range(ndepths - 1):
            rfo.atom_range_for_depth[i, 0] = depth_offsets[i]
            rfo.atom_range_for_depth[i, 1] = depth_offsets[i + 1]
        rfo.atom_range_for_depth[ndepths - 1, 0] = depth_offsets[-1]
        rfo.atom_range_for_depth[ndepths - 1, 1] = natoms

        subpath_roots = numpy.nonzero(is_subpath_root_ko)[0]
        root_depths = refold_atom_depth_ko[subpath_roots]
        for ii in range(ndepths):
            ii_roots = subpath_roots[root_depths == ii]
            cls.finalize_refold_indices(
                ii_roots, depth_offsets[ii], subpath_child_ko, rfo.ri2ki,
                rfo.ki2ri
            )

        assert numpy.all(rfo.ri2ki != -1)
        assert numpy.all(rfo.ki2ri != -1)

        rfo.is_subpath_root[:] = is_subpath_root_ko[rfo.ri2ki]
        rfo.non_subpath_parent[rfo.is_subpath_root] = \
            rfo.ki2ri[parent_ko[rfo.ri2ki][rfo.is_subpath_root]]
        rfo.non_subpath_parent[0] = -1

        return refold_ordering

    @staticmethod
    @numba.jit(nopython=True)
    def finalize_refold_indices(
            roots, depth_offset, subpath_child_ko, ri2ki, ki2ri
    ):
        count = depth_offset
        for root in roots:
            nextatom = root
            while nextatom != -1:
                ri2ki[count] = nextatom
                ki2ri[nextatom] = count
                nextatom = subpath_child_ko[nextatom]
                count += 1


@attr.s(auto_attribs=True)
class DerivsumOrdering:
    ki2dsi: numpy.ndarray
    dsi2ki: numpy.ndarray

    is_leaf: numpy.ndarray
    non_path_children: numpy.ndarray
    atom_range_for_depth: numpy.ndarray

    @classmethod
    def determine(
            cls,
            natoms,
            parent_ko,
            derivsum_path_depth_ko,
            subpath_length_ko,
            is_subpath_root_ko,
            is_subpath_leaf_ko,
            non_path_children_ko,
    ):
        n_derivsum_depths = derivsum_path_depth_ko[0] + 1

        dso = derivsum_ordering = cls(
            ki2dsi=numpy.full((natoms), -1, dtype="int32"),
            dsi2ki=numpy.full((natoms), -1, dtype="int32"),
            is_leaf=numpy.full((natoms), False, dtype="bool"),
            non_path_children=numpy.full_like(
                non_path_children_ko,
                -1,
                dtype="int32",
            ),
            atom_range_for_depth=numpy.full(
                (n_derivsum_depths, 2),
                -1,
                "int32",
            ),
        )

        leaf_path_depths = derivsum_path_depth_ko[is_subpath_leaf_ko]
        leaf_path_lengths = subpath_length_ko[is_subpath_leaf_ko]

        depth_offsets = numpy.zeros((n_derivsum_depths), dtype="int32")
        numpy.add.at(depth_offsets, leaf_path_depths, leaf_path_lengths)
        depth_offsets[1:] = numpy.cumsum(depth_offsets)[:-1]
        depth_offsets[0] = 0

        for ii in range(n_derivsum_depths - 1):
            dso.atom_range_for_depth[ii, 0] = depth_offsets[ii]
            dso.atom_range_for_depth[ii, 1] = depth_offsets[ii + 1]

        dso.atom_range_for_depth[n_derivsum_depths - 1, 0] = \
             depth_offsets[n_derivsum_depths - 1]
        dso.atom_range_for_depth[n_derivsum_depths - 1, 1] = natoms

        derivsum_leaves = numpy.nonzero(is_subpath_leaf_ko)[0]
        for ii in range(n_derivsum_depths):
            ii_leaves = derivsum_leaves[leaf_path_depths == ii]
            cls.finalize_derivsum_indices(
                leaves=ii_leaves,
                start_ind=depth_offsets[ii],
                parent=parent_ko,
                is_root=is_subpath_root_ko,
                ki2dsi=dso.ki2dsi,
                dsi2ki=dso.dsi2ki,
            )

        assert numpy.all(dso.ki2dsi != -1)
        assert numpy.all(dso.dsi2ki != -1)

        for ii in range(non_path_children_ko.shape[1]):
            child_exists = non_path_children_ko[:, ii] != -1
            dso.non_path_children[child_exists, ii] = dso.ki2dsi[
                non_path_children_ko[child_exists, ii]
            ]
        # now all the identies of the children have been remapped, but they
        # are still in kintree order; so reorder them to derivsum order.
        dso.non_path_children[:] = dso.non_path_children[dso.dsi2ki]
        dso.is_leaf[:] = is_subpath_leaf_ko[dso.dsi2ki]

        return derivsum_ordering

    @staticmethod
    @numba.jit(nopython=True)
    def finalize_derivsum_indices(
            leaves, start_ind, parent, is_root, ki2dsi, dsi2ki
    ):
        count = start_ind
        for leaf in leaves:
            nextatom = leaf
            while True:
                dsi2ki[count] = nextatom
                ki2dsi[nextatom] = count
                count += 1
                if is_root[nextatom]:
                    break
                nextatom = parent[nextatom]


@cuda.jit
def reorder_starting_hts(natoms, hts_ko, hts_ro, ki2ri):
    pos = cuda.grid(1)
    if pos < natoms:
        ri = ki2ri[pos]
        for i in range(12):
            hts_ro[ri, i] = hts_ko[pos, i // 4, i % 4]


@cuda.jit(device=True)
def reorder_starting_hts_256(hts_ko, hts_ro, ki2ri):
    natoms = hts_ko.shape[0]
    pos = cuda.grid(1)
    niters = (natoms - 1) // 256 + 1
    for ii in range(niters):
        which = pos + 256 * ii
        if which < natoms:
            ri = ki2ri[which]
            for i in range(12):
                hts_ro[ri, i] = hts_ko[which, i // 4, i % 4]


@cuda.jit
def reorder_final_hts(natoms, hts_ko, hts_ro, ki2ri):
    pos = cuda.grid(1)
    if pos < natoms:
        ri = ki2ri[pos]
        for i in range(12):
            hts_ko[pos, i // 4, i % 4] = hts_ro[ri, i]


@cuda.jit(device=True)
def reorder_final_hts_256(hts_ko, hts_ro, ki2ri):
    natoms = hts_ko.shape[0]
    pos = cuda.grid(1)
    niters = (natoms - 1) // 256 + 1
    for ii in range(niters):
        which = pos + 256 * ii
        if which < natoms:
            ri = ki2ri[which]
            for i in range(12):
                hts_ko[which, i // 4, i % 4] = hts_ro[ri, i]


@cuda.jit(device=True)
def identity_ht():
    one = numba.float32(1.0)
    zero = numba.float32(0.0)
    return (
        one, zero, zero, zero, zero, one, zero, zero, zero, zero, one, zero
    )


@cuda.jit(device=True)
def ht_load(hts, pos):
    v0 = hts[pos, 0]
    v1 = hts[pos, 1]
    v2 = hts[pos, 2]
    v3 = hts[pos, 3]
    v4 = hts[pos, 4]
    v5 = hts[pos, 5]
    v6 = hts[pos, 6]
    v7 = hts[pos, 7]
    v8 = hts[pos, 8]
    v9 = hts[pos, 9]
    v10 = hts[pos, 10]
    v11 = hts[pos, 11]
    return (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)


@cuda.jit(device=True)
def ht_load_from_shared(hts, pos):
    v0 = hts[0, pos]
    v1 = hts[1, pos]
    v2 = hts[2, pos]
    v3 = hts[3, pos]
    v4 = hts[4, pos]
    v5 = hts[5, pos]
    v6 = hts[6, pos]
    v7 = hts[7, pos]
    v8 = hts[8, pos]
    v9 = hts[9, pos]
    v10 = hts[10, pos]
    v11 = hts[11, pos]
    return (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)


@cuda.jit(device=True)
def ht_load_from_nx4x4(hts, pos):
    v0 = hts[pos, 0, 0]
    v1 = hts[pos, 0, 1]
    v2 = hts[pos, 0, 2]
    v3 = hts[pos, 0, 3]
    v4 = hts[pos, 1, 0]
    v5 = hts[pos, 1, 1]
    v6 = hts[pos, 1, 2]
    v7 = hts[pos, 1, 3]
    v8 = hts[pos, 2, 0]
    v9 = hts[pos, 2, 1]
    v10 = hts[pos, 2, 2]
    v11 = hts[pos, 2, 3]
    return (v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11)


@cuda.jit(device=True)
def ht_save(shared_hts, pos, ht):
    for i in range(12):
        shared_hts[pos, i] = ht[i]


@cuda.jit(device=True)
def ht_save_to_shared(shared_hts, pos, ht):
    for i in range(12):
        shared_hts[i, pos] = ht[i]


@cuda.jit(device=True)
def ht_save_to_n_x_12(hts, pos, ht):
    for i in range(12):
        hts[pos, i] = ht[i]


@cuda.jit(device=True)
def ht_save_to_nx4x4(hts, pos, ht):
    for i in range(12):
        hts[pos, i // 4, i % 4] = ht[i]


@cuda.jit(device=True)
def ht_multiply(ht1, ht2):

    r0 = ht1[0] * ht2[0] + ht1[1] * ht2[4] + ht1[2] * ht2[8]
    r1 = ht1[0] * ht2[1] + ht1[1] * ht2[5] + ht1[2] * ht2[9]
    r2 = ht1[0] * ht2[2] + ht1[1] * ht2[6] + ht1[2] * ht2[10]
    r3 = ht1[0] * ht2[3] + ht1[1] * ht2[7] + ht1[2] * ht2[11] + ht1[3]

    r4 = ht1[4] * ht2[0] + ht1[5] * ht2[4] + ht1[6] * ht2[8]
    r5 = ht1[4] * ht2[1] + ht1[5] * ht2[5] + ht1[6] * ht2[9]
    r6 = ht1[4] * ht2[2] + ht1[5] * ht2[6] + ht1[6] * ht2[10]
    r7 = ht1[4] * ht2[3] + ht1[5] * ht2[7] + ht1[6] * ht2[11] + ht1[7]

    r8 = ht1[8] * ht2[0] + ht1[9] * ht2[4] + ht1[10] * ht2[8]
    r9 = ht1[8] * ht2[1] + ht1[9] * ht2[5] + ht1[10] * ht2[9]
    r10 = ht1[8] * ht2[2] + ht1[9] * ht2[6] + ht1[10] * ht2[10]
    r11 = ht1[8] * ht2[3] + ht1[9] * ht2[7] + ht1[10] * ht2[11] + ht1[11]

    return (r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11)


@cuda.jit
def segscan_ht_intervals_one_thread_block(
        hts_ko, ri2ki, is_root, parent_ind, atom_ranges
):
    # this should be executed as a single thread block with nthreads = 256

    #reorder_starting_hts_256(hts_ko, hts, ki2ri)
    #cuda.syncthreads()

    shared_hts = cuda.shared.array((12, 256), numba.float64)
    shared_is_root = cuda.shared.array((256), numba.int32)

    pos = cuda.grid(1)

    for depth in range(atom_ranges.shape[0]):
        start = atom_ranges[depth, 0]
        end = atom_ranges[depth, 1]

        niters = (end - start - 1) // 256 + 1
        carry_ht = identity_ht()
        carry_is_root = False
        for ii in range(niters):
            ii_ind = ii * 256 + start + pos
            ki = ri2ki[ii_ind]
            #load data into shared memory
            if ii_ind < end:
                for jj in range(12):
                    shared_hts[jj, pos] = hts_ko[ki, jj // 4, jj % 4]
                shared_is_root[pos] = is_root[ii_ind]
                myht = ht_load_from_shared(shared_hts, pos)
                parent = parent_ind[ii_ind]
                htchanged = False
                if parent != -1:
                    parent_ki = ri2ki[parent]
                    parent_ht = ht_load_from_nx4x4(hts_ko, parent_ki)
                    myht = ht_multiply(parent_ht, myht)
                    htchanged = True
                myroot = shared_is_root[pos]
                if pos == 0 and not myroot:
                    myht = ht_multiply(carry_ht, myht)
                    myroot |= carry_is_root
                    shared_is_root[0] = myroot
                    htchanged = True
                if htchanged:
                    ht_save_to_shared(shared_hts, pos, myht)
            cuda.syncthreads()

            # begin segmented scan on this section
            offset = 1
            for jj in range(8):  #log2(256) == 8
                if pos >= offset and ii_ind < end:
                    prev_ht = ht_load_from_shared(shared_hts, pos - offset)
                    prev_root = shared_is_root[pos - offset]
                cuda.syncthreads()
                if pos >= offset and ii_ind < end:
                    if not myroot:
                        myht = ht_multiply(prev_ht, myht)
                        myroot |= prev_root
                        ht_save_to_shared(shared_hts, pos, myht)
                        shared_is_root[pos] = myroot
                offset *= 2
                cuda.syncthreads()

            # write the shared hts to global memory
            if ii_ind < end:
                ht_save_to_nx4x4(hts_ko, ki, myht)

            # save the carry
            if pos == 0:
                carry_ht = ht_load_from_shared(shared_hts, 255)
                carry_is_root = shared_is_root[255]

            cuda.syncthreads()

    #reorder_final_hts_256(hts_ko, hts, ki2ri)


@cuda.jit(device=True)
def ht_multiply_prev_and_store(pos, offset, myht, shared_hts):
    prevht = ht_load_from_shared(shared_hts, pos - offset)

    myht = ht_multiply(prevht, myht)
    ht_save_to_shared(shared_hts, pos, myht)

    return myht


@cuda.jit(device=True)
def warp_segscan_hts1(
        pos, warp_id, ri2ki, hts_ko, is_root, parent_ind, carry_ht, shared_hts,
        shared_is_root, int_hts, int_is_root, start, end
):
    ht_ind = start + pos
    lane = pos & 31
    if ht_ind < end:
        shared_is_root[pos] = is_root[ht_ind]

    warp_first = warp_id << 5  # ie warp_id * 32; is this faster than multiplication?
    warp_is_open = shared_is_root[warp_first] == 0

    myht = identity_ht()

    if shared_is_root[pos]:
        shared_is_root[pos] = lane

    # now compute mindex by performing a scan on shared_is_root
    # using "max" as the associative operator
    if lane >= 1:
        shared_is_root[pos] = max(shared_is_root[pos - 1], shared_is_root[pos])
    if lane >= 2:
        shared_is_root[pos] = max(shared_is_root[pos - 2], shared_is_root[pos])
    if lane >= 4:
        shared_is_root[pos] = max(shared_is_root[pos - 4], shared_is_root[pos])
    if lane >= 8:
        shared_is_root[pos] = max(shared_is_root[pos - 8], shared_is_root[pos])
    if lane >= 16:
        shared_is_root[pos] = max(
            shared_is_root[pos - 16], shared_is_root[pos]
        )

    mindex = shared_is_root[pos]
    ki = -1

    # pull down the hts from global memory into shared memory, and then
    # into thread-local memory. Then integrate the parent's HT into root
    # nodes (i.e. nodes whose parent is not listed as -1)
    if ht_ind < end:
        ki = ri2ki[ht_ind]
        for jj in range(12):
            shared_hts[jj, pos] = hts_ko[ki, jj // 4, jj % 4]
        myht = ht_load_from_shared(shared_hts, pos)
        parent = parent_ind[ht_ind]
        htchanged = False
        if parent != -1:
            parent_ki = ri2ki[parent]
            parent_ht = ht_load_from_nx4x4(hts_ko, parent_ki)
            myht = ht_multiply(parent_ht, myht)
            htchanged = True
        if pos == 0 and warp_is_open:
            myht = ht_multiply(carry_ht, myht)
            htchanged = True
        if htchanged:
            ht_save_to_shared(shared_hts, pos, myht)

        # now run segmented scan, unrolling the traditional for loop (does it really save time?)
        # no synchronization necessary for intra-warp scans since these threads are in
        # guaranteed lock sttep
        if lane >= mindex + 1:
            myht = ht_multiply_prev_and_store(pos, 1, myht, shared_hts)
        if lane >= mindex + 2:
            myht = ht_multiply_prev_and_store(pos, 2, myht, shared_hts)
        if lane >= mindex + 4:
            myht = ht_multiply_prev_and_store(pos, 4, myht, shared_hts)
        if lane >= mindex + 8:
            myht = ht_multiply_prev_and_store(pos, 8, myht, shared_hts)
        if lane >= mindex + 16:
            myht = ht_multiply_prev_and_store(pos, 16, myht, shared_hts)

    if lane == 31:
        # now lets write out the intermediate results
        ht_save_to_shared(int_hts, warp_id, myht)
        int_is_root[warp_id] = mindex != 0 or not warp_is_open

    # for the third stage of this intra-block scan, record whether this
    # thread should accumulate the scanned intermediate HT from stage 2
    will_accumulate = warp_is_open and mindex == 0

    return ki, myht, will_accumulate


@cuda.jit(device=True)
def warp_segscan_hts2(pos, int_hts, int_is_root):
    """Now we'll perform a rapid inclusive scan on the int_hts, to merge
    the scanned HTs of all the warps in the thread block. The only threads
    that ought to execute this code are threads 0-7"""

    myht = ht_load_from_shared(int_hts, pos)

    # scan the isroot flags to compute the mindex
    if int_is_root[pos]:
        int_is_root[pos] = pos
    else:
        int_is_root[pos] = 0

    if pos >= 1:
        int_is_root[pos] = max(int_is_root[pos - 1], int_is_root[pos])
    if pos >= 2:
        int_is_root[pos] = max(int_is_root[pos - 2], int_is_root[pos])
    if pos >= 4:
        int_is_root[pos] = max(int_is_root[pos - 4], int_is_root[pos])
    mindex = int_is_root[pos]

    # now scan, unrolling the traditional for loop (does it really save time?)
    if pos >= mindex + 1:
        myht = ht_multiply_prev_and_store(pos, 1, myht, int_hts)
    if pos >= mindex + 2:
        myht = ht_multiply_prev_and_store(pos, 2, myht, int_hts)
    if pos >= mindex + 4:
        myht = ht_multiply_prev_and_store(pos, 4, myht, int_hts)


@cuda.jit
def segscan_ht_intervals_one_thread_block2(
        hts_ko, ri2ki, is_root, parent_ind, atom_ranges
):
    # this should be executed as a single thread block with nthreads = 256
    # The idea behind this version is to run within-warp scans to get
    # the carry; then to run a quick one-warp version of scan on the carries;
    # and finally, to apply the carries back to the original warp sections.

    pos = cuda.grid(1)
    warp_id = pos >> 5  # bit shift by 5 because warp size is 32

    #reorder_starting_hts_256(hts_ko, hts, ki2ri)
    #cuda.syncthreads()

    shared_hts = cuda.shared.array((12, 256), numba.float64)
    shared_is_root = cuda.shared.array((256), numba.int32)

    shared_intermediate_hts = cuda.shared.array((12, 8), numba.float64)
    shared_intermediate_is_root = cuda.shared.array((12), numba.int32)

    pos = cuda.grid(1)

    for depth in range(atom_ranges.shape[0]):
        start = atom_ranges[depth, 0]
        end = atom_ranges[depth, 1]

        niters = (end - start - 1) // 256 + 1
        carry_ht = identity_ht()
        for ii in range(niters):

            ii_start = start + ii * 256

            # stage 1:
            ki, myht, will_accumulate = warp_segscan_hts1(
                pos, warp_id, ri2ki, hts_ko, is_root, parent_ind, carry_ht,
                shared_hts, shared_is_root, shared_intermediate_hts,
                shared_intermediate_is_root, ii_start, end
            )
            cuda.syncthreads()

            # stage 2:
            if pos < 8:
                warp_segscan_hts2(
                    pos, shared_intermediate_hts, shared_intermediate_is_root
                )
            cuda.syncthreads()

            # stage 3:
            if will_accumulate and warp_id != 0 and ii_start + pos < end:
                prev_ht = ht_load_from_shared(
                    shared_intermediate_hts, warp_id - 1
                )
                myht = ht_multiply(prev_ht, myht)
            if ii_start + pos < end:
                ht_save_to_nx4x4(hts_ko, ki, myht)
                #for jj in range(12):
                #    hts_ko[ ki, jj // 4, jj % 4 ] = myht[jj]
            ht_save_to_shared(shared_hts, pos, myht)
            cuda.syncthreads()

            # save the carry
            if pos == 0:
                carry_ht = ht_load_from_shared(shared_hts, 255)

            cuda.syncthreads()


def get_devicendarray(t):
    import ctypes
    '''Convert a device-allocated pytorch tensor into a numba DeviceNDArray'''
    if t.type() == 'torch.cuda.FloatTensor':
        ctx = cuda.cudadrv.driver.driver.get_context()
        mp = cuda.cudadrv.driver.MemoryPointer(
            ctx, ctypes.c_ulong(t.data_ptr()),
            t.numel() * 4
        )
        return cuda.cudadrv.devicearray.DeviceNDArray(
            t.size(), [i * 4 for i in t.stride()],
            numpy.dtype('float32'),
            gpu_data=mp,
            stream=torch.cuda.current_stream().cuda_stream
        )
    elif t.type() == 'torch.cuda.DoubleTensor':
        ctx = cuda.cudadrv.driver.driver.get_context()
        mp = cuda.cudadrv.driver.MemoryPointer(
            ctx, ctypes.c_ulong(t.data_ptr()),
            t.numel() * 8
        )
        return cuda.cudadrv.devicearray.DeviceNDArray(
            t.size(), [i * 8 for i in t.stride()],
            numpy.dtype('float64'),
            gpu_data=mp,
            stream=torch.cuda.current_stream().cuda_stream
        )
    else:
        # We're using the numba cuda simulator; this will let us modify the underlying
        # numpy array in numba on the CPU. Neat!
        return t.numpy()


def segscan_hts_gpu(hts_ko, reordering):
    """Perform a series of segmented scan operations on the input homogeneous transforms
    to compute the coordinates (and coordinate frames) of all atoms in the structure.
    This version uses cuda.syncthreads() calls to ensure that there are no data race
    issues. For this reason, it can be safely run on the CPU using numba's CUDA simulator
    (activated by setting the environment variable NUMBA_ENABLE_CUDASIM=1)"""

    ro = reordering
    stream = cuda.stream()

    segscan_ht_intervals_one_thread_block[1, 256, stream](
        hts_ko, ro.ri2ki, ro.is_root_ro, ro.non_subpath_parent_ro,
        ro.refold_atom_ranges
    )


def warp_synchronous_segscan_hts_gpu(hts_ko, reordering):
    '''Perform a series of segmented scan operations on the input homogeneous transforms
    to compute the coordinates (and coordinate frames) of all atoms in the structure.
    Uses warp-synchronous programming to minimize the number of thread-block synchronization
    events. Warp-synchronous programming is no longer a good idea; it is deprecated-ish
    in CUDA-9 and warp synchronicity is not guaranteed on the Volta architecture. This
    version is also not faser than the other refold version(?!) which surprises me.'''
    ro = reordering
    stream = cuda.stream()

    segscan_ht_intervals_one_thread_block2[1, 256, stream](
        hts_ko, ro.ri2ki, ro.is_root_ro, ro.non_subpath_parent_ro,
        ro.refold_atom_ranges
    )


@numba.jit(nopython=True)
def mark_path_children_and_count_nonpath_children(
        natoms,
        parent_ko,
        subpath_child_ko,
        out_n_nonpath_children_ko,
        out_is_subpath_root_ko,
        out_is_subpath_leaf_ko,
):
    for ii in range(natoms - 1, -1, -1):
        ii_parent = parent_ko[ii]
        if ii == ii_parent:
            out_is_subpath_root_ko[ii] = True
        elif subpath_child_ko[ii_parent] != ii:
            out_n_nonpath_children_ko[ii_parent] += 1
            out_is_subpath_root_ko[ii] = True
        out_is_subpath_leaf_ko[ii] = subpath_child_ko[ii] == -1


@numba.jit(nopython=True)
def list_nonpath_children(
        natoms, is_subpath_root_ko, parent_ko, out_non_path_children_ko
):
    count_n_nonfirst_children = numpy.zeros((natoms), dtype=numpy.int32)
    for ii in range(natoms):
        if is_subpath_root_ko[ii]:
            ii_parent = parent_ko[ii]
            if ii_parent == ii:
                continue
            ii_child_ind = count_n_nonfirst_children[ii_parent]
            out_non_path_children_ko[ii_parent, ii_child_ind] = ii
            count_n_nonfirst_children[ii_parent] += 1


@numba.jit(nopython=True)
def find_derivsum_path_depths(
        natoms,
        subpath_child_ko,
        non_path_children_ko,
        is_subpath_root_ko,
        out_derivsum_path_depth_ko,
        out_subpath_length_ko,
):
    for ii in range(natoms - 1, -1, -1):
        # my depth is the larger of my first child's depth, or
        # my other children's laregest depth + 1
        ii_depth = 0
        ii_child = subpath_child_ko[ii]
        if ii_child != -1:
            ii_depth = out_derivsum_path_depth_ko[ii_child]
        for other_child in non_path_children_ko[ii, :]:
            if other_child == -1:
                continue
            other_child_depth = out_derivsum_path_depth_ko[other_child]
            if ii_depth < other_child_depth + 1:
                ii_depth = other_child_depth + 1
        out_derivsum_path_depth_ko[ii] = ii_depth

        # if this is the root of a derivsum path (remember, paths are summed
        # leaf to root), then visit all of the nodes on the path and mark them
        # with my depth. I'm not sure this is necessary
        if is_subpath_root_ko[ii]:
            next_node = subpath_child_ko[ii]
            path_length = 1
            leaf_node = ii
            while next_node != -1:
                leaf_node = next_node
                out_derivsum_path_depth_ko[next_node] = ii_depth
                next_node = subpath_child_ko[next_node]
                if next_node != -1:
                    leaf_node = next_node
                path_length += 1

            out_subpath_length_ko[ii] = path_length
            out_subpath_length_ko[leaf_node] = path_length


@cuda.jit(device=True)
def load_f1f2s(f1f2s, ind):
    v0 = f1f2s[ind, 0]
    v1 = f1f2s[ind, 1]
    v2 = f1f2s[ind, 2]
    v3 = f1f2s[ind, 3]
    v4 = f1f2s[ind, 4]
    v5 = f1f2s[ind, 5]
    return (v0, v1, v2, v3, v4, v5)


@cuda.jit(device=True)
def add_f1f2s(v1, v2):
    res0 = v1[0] + v2[0]
    res1 = v1[1] + v2[1]
    res2 = v1[2] + v2[2]
    res3 = v1[3] + v2[3]
    res4 = v1[4] + v2[4]
    res5 = v1[5] + v2[5]
    return (res0, res1, res2, res3, res4, res5)


@cuda.jit(device=True)
def save_f1f2s(f1f2s, ind, v):
    for i in range(6):
        f1f2s[ind, i] = v[i]


@cuda.jit(device=True)
def zero_f1f2s():
    zero = numba.float64(0.)
    return (zero, zero, zero, zero, zero, zero)


@cuda.jit
def reorder_starting_f1f2s(natoms, f1f2s_ko, f1f2s_dso, ki2dsi):
    pos = cuda.grid(1)
    if pos < natoms:
        dsi = ki2dsi[pos]
        for i in range(6):
            f1f2s_dso[dsi, i] = f1f2s_ko[pos, i]


@cuda.jit
def reorder_final_f1f2s(natoms, f1f2s_ko, f1f2s_dso, ki2dsi):
    pos = cuda.grid(1)
    if pos < natoms:
        dsi = ki2dsi[pos]
        for i in range(6):
            f1f2s_ko[pos, i] = f1f2s_dso[dsi, i]


@cuda.jit
def segscan_f1f2s_up_tree(
        f1f2s_dso, prior_children, is_leaf, derivsum_atom_ranges
):
    shared_f1f2s = cuda.shared.array((512, 6), numba.float64)
    shared_is_leaf = cuda.shared.array((512), numba.int32)
    pos = cuda.grid(1)

    for depth in range(derivsum_atom_ranges.shape[0]):
        start = derivsum_atom_ranges[depth, 0]
        end = derivsum_atom_ranges[depth, 1]

        niters = (end - start - 1) // 512 + 1
        carry_f1f2s = zero_f1f2s()
        carry_is_leaf = False
        for ii in range(niters):
            ii_ind = ii * 512 + start + pos
            if ii_ind < end:
                for jj in range(6):
                    # TO DO: minimize bank conflicts -- align memory reads
                    shared_f1f2s[pos, jj] = f1f2s_dso[ii_ind, jj]
                shared_is_leaf[pos] = is_leaf[ii_ind]
                myf1f2s = load_f1f2s(shared_f1f2s, pos)
                my_leaf = shared_is_leaf[pos]
                f1f2s_changed = False
                for jj in range(prior_children.shape[1]):
                    jj_child = prior_children[ii_ind, jj]
                    if jj_child != -1:
                        child_f1f2s = load_f1f2s(f1f2s_dso, jj_child)
                        myf1f2s = add_f1f2s(myf1f2s, child_f1f2s)
                        f1f2s_changed = True
                if pos == 0 and not my_leaf:
                    myf1f2s = add_f1f2s(carry_f1f2s, myf1f2s)
                    my_leaf |= carry_is_leaf
                    shared_is_leaf[0] = my_leaf
                    f1f2s_changed = True
                if f1f2s_changed:
                    save_f1f2s(shared_f1f2s, pos, myf1f2s)
            cuda.syncthreads()

            # begin segmented scan on this section
            offset = 1
            for jj in range(9):
                if pos >= offset and ii_ind < end:
                    prev_f1f2s = load_f1f2s(shared_f1f2s, pos - offset)
                    prev_leaf = shared_is_leaf[pos - offset]
                cuda.syncthreads()
                if pos >= offset and ii_ind < end:
                    if not my_leaf:
                        myf1f2s = add_f1f2s(myf1f2s, prev_f1f2s)
                        my_leaf |= prev_leaf
                        save_f1f2s(shared_f1f2s, pos, myf1f2s)
                        shared_is_leaf[pos] = my_leaf
                offset *= 2
                cuda.syncthreads()

            # write the f1f2s to global memory
            if ii_ind < end:
                save_f1f2s(f1f2s_dso, ii_ind, myf1f2s)

            # save the carry
            if pos == 0:
                carry_f1f2s = load_f1f2s(shared_f1f2s, 511)
                carry_is_leaf = shared_is_leaf[511]

            cuda.syncthreads()


def segscan_f1f2s_gpu(f1f2s_ko, reordering):
    # TO DO: handle f1 and f2 separately; segscan in-place (no reordering kernels)
    ro = reordering
    f1f2s_dso_d = cuda.device_array((ro.natoms, 6), dtype="float64")

    nblocks = (ro.natoms - 1) // 512 + 1
    reorder_starting_f1f2s[nblocks, 512](
        ro.natoms, f1f2s_ko, f1f2s_dso_d, ro.ki2dsi
    )

    segscan_f1f2s_up_tree[1, 512](
        f1f2s_dso_d, ro.non_path_children_dso, ro.is_leaf_dso,
        ro.derivsum_atom_ranges
    )

    reorder_final_f1f2s[nblocks, 512](
        ro.natoms, f1f2s_ko, f1f2s_dso_d, ro.ki2dsi
    )
