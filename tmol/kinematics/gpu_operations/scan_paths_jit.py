import numba
import numpy


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
    natoms, parent_ko, is_subpath_root_ko, out_refold_atom_depth_ko
):
    for ii in range(natoms):
        ii_parent = parent_ko[ii]
        ii_depth = out_refold_atom_depth_ko[ii_parent]
        if is_subpath_root_ko[ii] and ii_parent != ii:
            ii_depth += 1
        out_refold_atom_depth_ko[ii] = ii_depth
