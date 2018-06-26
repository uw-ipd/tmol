import attr

import numpy
import numba

from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray

from .forward_jit import segscan_ht_intervals_one_thread_block


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
