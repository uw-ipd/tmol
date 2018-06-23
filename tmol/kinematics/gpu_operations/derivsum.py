import attr

import numpy
import numba

from .derivsum_jit import (
    reorder_starting_f1f2s,
    segscan_f1f2s_up_tree,
    reorder_final_f1f2s,
)


def segscan_f1f2s_gpu(f1f2s_ko, reordering):
    # TO DO: handle f1 and f2 separately; segscan in-place (no reordering kernels)
    ro = reordering
    f1f2s_dso_d = numba.cuda.device_array((ro.natoms, 6), dtype="float64")

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
