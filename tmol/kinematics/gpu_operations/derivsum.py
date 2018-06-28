import attr

import numpy
import numba

from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray
from tmol.types.torch import Tensor

from tmol.utility.numba import as_cuda_array

from .scan_paths import PathPartitioning

from .derivsum_jit import (
    finalize_derivsum_indices,
    reorder_starting_f1f2s,
    segscan_f1f2s_up_tree,
    reorder_final_f1f2s,
)


@attr.s(auto_attribs=True)
class DerivsumOrdering(ValidateAttrs):
    # [natoms]
    ki2dsi: NDArray("i4")[:]
    dsi2ki: NDArray("i4")[:]
    is_leaf: NDArray("bool")[:]

    # [natoms, max_num_nonpath_children
    nonpath_children: NDArray("i4")[:, :]

    # [n_path_depths, 2]
    atom_range_for_depth: NDArray("i4")[:, 2]

    @classmethod
    def for_scan_paths(cls, scan_paths: PathPartitioning):
        natoms = len(scan_paths.parent)
        ndepths = scan_paths.subpath_depth_from_leaf[0] + 1

        # Determine the number of atoms present in paths at each path depth,
        # and the index ranges needed for each depth in the scan buffer.
        subpath_leaves = numpy.flatnonzero(scan_paths.is_subpath_leaf)
        subpath_depths_from_leaf = scan_paths.subpath_depth_from_leaf[
            subpath_leaves
        ]
        subpath_lengths = scan_paths.subpath_length[subpath_leaves]

        depth_offsets = numpy.zeros((ndepths), dtype="int32")
        numpy.add.at(depth_offsets, subpath_depths_from_leaf, subpath_lengths)
        depth_offsets[1:] = numpy.cumsum(depth_offsets)[:-1]
        depth_offsets[0] = 0

        atom_range_for_depth = numpy.full(
            (ndepths, 2),
            -1,
            dtype="int32",
        )

        for ii in range(ndepths - 1):
            atom_range_for_depth[ii, 0] = depth_offsets[ii]
            atom_range_for_depth[ii, 1] = depth_offsets[ii + 1]

        atom_range_for_depth[ndepths - 1, 0] = depth_offsets[ndepths - 1]
        atom_range_for_depth[ndepths - 1, 1] = natoms

        dsi2ki = numpy.full((natoms), -1, dtype="int32")
        ki2dsi = numpy.full((natoms), -1, dtype="int32")
        for ii in range(ndepths):
            ii_leaves = subpath_leaves[subpath_depths_from_leaf == ii]
            finalize_derivsum_indices(
                leaves=ii_leaves,
                start_ind=depth_offsets[ii],
                parent=scan_paths.parent,
                is_root=scan_paths.is_subpath_root,
                ki2dsi=ki2dsi,
                dsi2ki=dsi2ki,
            )

        assert numpy.all(ki2dsi != -1)
        assert numpy.all(dsi2ki != -1)

        return cls(
            dsi2ki=dsi2ki,
            ki2dsi=ki2dsi,
            is_leaf=scan_paths.is_subpath_leaf[dsi2ki],
            # First map nonpath children into dsi indices, then map
            # the nonpath children array into dsi ordering
            nonpath_children=numpy.where(
                scan_paths.nonpath_children != -1,
                ki2dsi[scan_paths.nonpath_children], -1
            )[dsi2ki],
            atom_range_for_depth=atom_range_for_depth
        )

    def segscan_f1f2s(
            self,
            f1f2s_kintree_ordering: Tensor("f8")[:, 6],
            inplace: bool = False
    ) -> Tensor("f8")[:, 6]:
        if not inplace:
            f1f2s_kintree_ordering = f1f2s_kintree_ordering.clone()

        # TODO: handle f1 and f2 separately; segscan in-place (no reordering kernels)
        natoms = len(f1f2s_kintree_ordering)
        assert natoms == len(self.dsi2ki)

        f1f2s_kintree_ordering_d = as_cuda_array(f1f2s_kintree_ordering)

        f1f2s_dso_d = numba.cuda.device_array((natoms, 6), dtype="float64")

        nblocks = (natoms - 1) // 512 + 1

        #TODO asford: isn't this just an indexing operation?
        reorder_starting_f1f2s[nblocks, 512](
            natoms,
            f1f2s_kintree_ordering_d,
            f1f2s_dso_d,
            self.ki2dsi,
        )

        segscan_f1f2s_up_tree[1, 512](
            f1f2s_dso_d,
            self.nonpath_children,
            self.is_leaf,
            self.derivsum_atom_ranges,
        )

        #TODO asford: isn't this just an indexing operation?
        reorder_final_f1f2s[nblocks, 512](
            natoms,
            f1f2s_kintree_ordering_d,
            f1f2s_dso_d,
            self.ki2dsi,
        )

        return f1f2s_kintree_ordering
