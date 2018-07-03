import attr

from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from ..datatypes import KinTree

from .scan_paths import PathPartitioning
from .forward import RefoldOrdering
from .derivsum import DerivsumOrdering


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

    The same set of paths is used for both the refold algorithm and
    the derivative summation; the refold algorithm starts at path
    roots and multiplies homogeneous transforms towards the leaves.
    The derivative summation algorithm starts at the leaves and sums
    upwards towards the roots.

    In order to divide the tree into these paths, this class constructs
    two reorderings of the atoms: a `refold_ordering` using refold
    indices (ri) and a `derivsum_ordering` using derivsum indices (di).
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

        refold_ordering = RefoldOrdering.for_scan_paths(scan_paths)
        derivsum_ordering = DerivsumOrdering.for_scan_paths(scan_paths)

        return cls(
            natoms=len(kintree),
            scan_paths=scan_paths,
            refold_ordering=refold_ordering,
            derivsum_ordering=derivsum_ordering,
        )


__all__ = (
    GPUKinTreeReordering, PathPartitioning, RefoldOrdering, DerivsumOrdering
)
