import attr

from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from ..datatypes import KinTree

from .scan_paths import PathPartitioning
from .forward import segscan_hts_gpu, RefoldOrdering
from .derivsum import segscan_f1f2s_gpu, DerivsumOrdering


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
    scan_paths: PathPartitioning

    # Kinematic refold (forward) data arrays
    refold_ordering: RefoldOrdering

    # Derivative summation (backward) data arrays
    derivsum_ordering: DerivsumOrdering

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
    def calculate_from_kintree(cls, kintree: KinTree):
        """Setup for operations over KinTree.

        `device` for gpu array is inferred from kintree tensor device.
        """

        scan_paths = PathPartitioning.minimum_subpath_depth(
            kintree.parent.cpu().numpy()
        )

        ### Map from paths and to forward kinematic refold indices
        refold_ordering = RefoldOrdering.for_scan_paths(scan_paths)

        derivsum_ordering = DerivsumOrdering.for_scan_paths(scan_paths)

        return cls(
            natoms=len(kintree),
            scan_paths=scan_paths,
            refold_ordering=refold_ordering,
            derivsum_ordering=derivsum_ordering,
        )

        # # Kinematic refold (forward) data arrays
        # ki2ri=to_cuda_device(refold_ordering.ki2ri),
        # ri2ki=to_cuda_device(refold_ordering.ri2ki),
        # non_subpath_parent_ro=to_cuda_device(
        #     refold_ordering.non_subpath_parent
        # ),
        # is_root_ro=to_cuda_device(refold_ordering.is_subpath_root),
        # refold_atom_ranges=to_cuda_device(
        #     refold_ordering.atom_range_for_depth
        # ),

        # # Derivative summation (backward) data arrays
        # ki2dsi=to_cuda_device(derivsum_ordering.ki2dsi),
        # dsi2ki=to_cuda_device(derivsum_ordering.dsi2ki),
        # is_leaf_dso=to_cuda_device(derivsum_ordering.is_leaf),
        # non_path_children_dso=to_cuda_device(
        #     derivsum_ordering.non_path_children
        # ),
        # derivsum_atom_ranges=to_cuda_device(
        #     derivsum_ordering.atom_range_for_depth
        # ),
        # )


__all__ = (GPUKinTreeReordering, segscan_hts_gpu, segscan_f1f2s_gpu)
