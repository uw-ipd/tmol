import numpy
import numba
import numba.cuda

from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray
from tmol.types.torch import Tensor

from tmol.utility.numba import as_cuda_array
from tmol.utility.reactive import reactive_attrs, reactive_property

from .scan_paths import PathPartitioning

from . import derivsum_jit


@reactive_attrs(auto_attribs=True, slots=True, frozen=True)
class DerivsumOrdering(ValidateAttrs):
    # [natoms]
    ki2dsi: NDArray("i4")[:]
    dsi2ki: NDArray("i4")[:]
    is_leaf: NDArray("bool")[:]

    # [natoms, max_num_nonpath_children
    nonpath_children: NDArray("i4")[:, :]

    # [n_path_depths, 2]
    atom_range_for_depth: NDArray("i4")[:, 2]

    # Cached device arrays, derived from cpu ordering arrays.
    @reactive_property
    def dsi2ki_d(dsi2ki):
        return numba.cuda.to_device(dsi2ki)

    @reactive_property
    def is_leaf_d(is_leaf):
        return numba.cuda.to_device(is_leaf)

    @reactive_property
    def nonpath_children_d(nonpath_children):
        return numba.cuda.to_device(nonpath_children)

    @reactive_property
    def atom_range_for_depth_d(atom_range_for_depth):
        return numba.cuda.to_device(atom_range_for_depth)

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
            derivsum_jit.finalize_derivsum_indices(
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

        natoms = len(f1f2s_kintree_ordering)
        assert natoms == len(self.dsi2ki)

        derivsum_jit.F1F2Scan.segscan_by_generation(
            64,
            as_cuda_array(f1f2s_kintree_ordering),
            self.dsi2ki_d,
            self.is_leaf_d,
            self.nonpath_children_d,
            self.atom_range_for_depth_d,
        )

        return f1f2s_kintree_ordering
