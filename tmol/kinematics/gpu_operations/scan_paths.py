from typing import Optional

import attr

import numpy
import numba

from ..datatypes import KinTree
from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from numba.cuda import to_device as to_cuda_device

from .scan_paths_jit import (
    mark_path_children_and_count_nonpath_children,
    list_nonpath_children,
    find_derivsum_path_depths,
    compute_branching_factor,
    find_refold_path_depths,
)
from .forward import RefoldOrdering
from .derivsum import DerivsumOrdering

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

    kintree_cache_key = "__GPUKinTreeReordering_cache__"

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

    @classmethod
    @validate_args
    def for_kintree(cls, kintree):
        """Calculate and cache refold ordering over kintree

        KinTree data structure is frozen; so it is safe to cache the gpu scan
        ordering for a single object. Store as a private property of the input
        kintree, lifetime of the cache will then be managed via the target
        object.
        ."""

        if not hasattr(kintree, cls.kintree_cache_key):
            object.__setattr__(
                kintree,
                cls.kintree_cache_key,
                cls.calculate_from_kintree(kintree),
            )

        return getattr(kintree, cls.kintree_cache_key)

    @classmethod
    @validate_args
    def calculate_from_kintree(cls, kintree: KinTree, device=None):
        """Setup for operations over KinTree.

        `device` for gpu array is inferred from kintree tensor device.
        """
        if device is None:
            device = kintree.parent.device

        if device.type != "cuda":
            raise ValueError(
                f"GPUKinTreeReordering not supported for non-cuda devices."
                f" device: {device}"
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
