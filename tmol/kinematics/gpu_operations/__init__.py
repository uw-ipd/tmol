import attr

from tmol.types.attrs import ValidateAttrs
from tmol.types.functional import validate_args

from ..datatypes import KinTree

from .scan_paths import PathPartitioning
from .forward import RefoldOrdering
from .derivsum import DerivsumOrdering


@attr.s(auto_attribs=True, frozen=True)
class GPUKinTreeReordering(ValidateAttrs):
    """Scan plans for parallel kinematic operations.

    The GPUKinTreeReordering divides the tree into a set of paths. Along
    each path is a continuous chain of atoms that either (1) require their
    coordinate frames computed as a cumulative product of homogeneous
    transforms for the coordinate update algorithm, or (2) require the
    cumulative sum of their f1f2 vectors for the derivative calculation
    algorithm. In both cases, these paths can be processed efficiently
    on the GPU using an algorithm called "scan" and batches of these paths
    can be processed at once in a variant called "segmented scan."

    The same set of paths is used for both the refold algorithm and
    the derivative summation; the refold algorithm starts at path
    roots and multiplies homogeneous transforms towards the leaves.
    The derivative summation algorithm starts at the leaves and sums
    upwards towards the roots.

    To accomplish this, the GPUKinTreeReordering class reorders the atoms from
    the original KinTree order ("ko") where atoms are known by their
    kintree-index ("ki") into 1) their refold order ("ro") where atoms are
    known by their refold index ("ri") and 2) their deriv-sum order ("dso")
    where atoms are known by their deriv-sum index.

    Each scan operation is performed as a series of depth-based "generations",
    in which the scans at generation n are *only* dependent on the results of
    scans in generations [0...n-1] inclusive. In the forward-scan generations
    are ordered from the kinematic root, and scan segments begin with a value
    derived from the result of a generation [0...n-1] scan. In the
    backward-scan generations are ordered from kinematic leaves, and scan
    segments pull summation results from multiple generation [0...n-1] scans.

    For further details on parallel segmented scan operations see:

    * Mark Harris, "Parallel Prefix Sum with CUDA."
      GPU Gems 3, Nvidia Corporation
      https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
      http://developer.download.nvidia.com/
      compute/cuda/2_2/sdk/website/projects/scan/doc/scan.pdf

    * Sengupta, Shubhabrata, et al. "Scan primitives for GPU computing."
      Graphics hardware. Vol. 2007. 2007.
      http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Sengupta07.pdf

    * Sean Baxter, "moderngpu"
      http://moderngpu.github.io/moderngpu
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
    "GPUKinTreeReordering", "PathPartitioning", "RefoldOrdering",
    "DerivsumOrdering"
)
