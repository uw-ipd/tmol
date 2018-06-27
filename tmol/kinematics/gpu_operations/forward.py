import attr

import numpy
import numba

from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray

from .forward_jit import (
    segscan_ht_intervals_one_thread_block, finalize_refold_indices
)
from .scan_paths import PathPartitioning


def segscan_hts_gpu(hts_ko, reordering):
    """Perform a series of segmented scan operations on the input homogeneous transforms
    to compute the coordinates (and coordinate frames) of all atoms in the structure.
    This version uses cuda.syncthreads() calls to ensure that there are no data race
    issues. For this reason, it can be safely run on the CPU using numba's CUDA simulator
    (activated by setting the environment variable NUMBA_ENABLE_CUDASIM=1)"""

    ro = reordering
    stream = numba.cuda.stream()

    segscan_ht_intervals_one_thread_block[1, 256, stream](
        hts_ko, ro.ri2ki, ro.is_root_ro, ro.non_subpath_parent_ro,
        ro.refold_atom_ranges
    )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RefoldOrdering(ValidateAttrs):

    # [natoms]
    ri2ki: NDArray("i4")[:]
    ki2ri: NDArray("i4")[:]

    is_subpath_root: NDArray("bool")[:]
    non_subpath_parent: NDArray("i4")[:]

    # [n_path_depths, 2]
    atom_range_for_depth: NDArray("i4")[:, 2]

    @classmethod
    def for_scan_paths(cls, scan_paths: PathPartitioning):
        natoms = len(scan_paths.parent)
        ndepths = scan_paths.subpath_depth_from_root.max() + 1

        # Determine the number of atoms present in paths at each path depth,
        # and the index ranges needed for each depth in the scan buffer.
        subpath_roots = numpy.flatnonzero(scan_paths.is_subpath_root)
        subpath_depths_from_root = scan_paths.subpath_depth_from_root[
            subpath_roots
        ]
        subpath_lengths = scan_paths.subpath_length[subpath_roots]

        depth_offsets = numpy.zeros((ndepths), dtype="int32")
        numpy.add.at(
            depth_offsets,
            subpath_depths_from_root,
            subpath_lengths,
        )
        depth_offsets[1:] = numpy.cumsum(depth_offsets)[:-1]
        depth_offsets[0] = 0

        atom_range_for_depth = numpy.full(
            (ndepths, 2),
            -1,
            dtype="int32",
        )
        for i in range(ndepths - 1):
            atom_range_for_depth[i, 0] = depth_offsets[i]
            atom_range_for_depth[i, 1] = depth_offsets[i + 1]
        atom_range_for_depth[ndepths - 1, 0] = depth_offsets[-1]
        atom_range_for_depth[ndepths - 1, 1] = natoms

        # Pack paths into scan buffer, path contiguous and grouped by depth.
        ri2ki = numpy.full((natoms), -1, dtype="int32")
        ki2ri = numpy.full((natoms), -1, dtype="int32")

        for ii in range(ndepths):
            ii_roots = subpath_roots[subpath_depths_from_root == ii]
            finalize_refold_indices(
                ii_roots,
                depth_offsets[ii],
                scan_paths.subpath_child,
                ri2ki,
                ki2ri,
            )

        assert numpy.all(ri2ki != -1)
        assert numpy.all(ki2ri != -1)

        return cls(
            ri2ki=ri2ki,
            ki2ri=ki2ri,
            atom_range_for_depth=atom_range_for_depth,
            is_subpath_root=scan_paths.is_subpath_root[ri2ki],
            non_subpath_parent=numpy.where(
                scan_paths.is_subpath_root[ri2ki],
                ki2ri[scan_paths.parent[ri2ki]],
                -1,
            ),
        )
