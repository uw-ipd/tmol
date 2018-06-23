from typing import Optional

import attr

import numpy
import numba
import torch

from ..datatypes import KinTree
from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

# Assign rather than import to respect dynamic numba cuda layout
from numba.cuda import to_device as to_cuda_device
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
            ki2ri=to_cuda_device(refold_ordering.ki2ri),
            ri2ki=to_cuda_device(refold_ordering.ri2ki),
            non_subpath_parent_ro=to_cuda_device(
                refold_ordering.non_subpath_parent
            ),
            is_root_ro=to_cuda_device(refold_ordering.is_subpath_root),
            refold_atom_ranges=to_cuda_device(
                refold_ordering.atom_range_for_depth
            ),

            # Derivative summation (backward) data arrays
            ki2dsi=to_cuda_device(derivsum_ordering.ki2dsi),
            dsi2ki=to_cuda_device(derivsum_ordering.dsi2ki),
            is_leaf_dso=to_cuda_device(derivsum_ordering.is_leaf),
            non_path_children_dso=to_cuda_device(
                derivsum_ordering.non_path_children
            ),
            derivsum_atom_ranges=to_cuda_device(
                derivsum_ordering.atom_range_for_depth
            ),
        )

    def active(self):
        return self.natoms != 0


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
