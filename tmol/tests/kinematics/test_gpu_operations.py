from tmol.kinematics.gpu_operations.scan_paths import GPUKinTreeReordering, RefoldOrdering, DerivsumOrdering


def test_gpu_refold_data_construction(ubq_kintree):
    kintree = ubq_kintree

    ### Otherwise test the derived ordering
    o: GPUKinTreeReordering = GPUKinTreeReordering.calculate_from_kintree(
        kintree
    )

    # Extract path data from tree reordering.
    # natoms = ordering.natoms
    # subpath_child_ko = ordering.subpath_child_ko
    # ki2ri = ordering.ki2ri
    # dsi2ki = ordering.dsi2ki
    # parent_ko = kintree.parent
    # non_subpath_parent_ro = ordering.non_subpath_parent_ro
    # subpath_child_ko = ordering.subpath_child_ko
    # non_path_children_ko = ordering.non_path_children_ko
    # non_path_children_dso = ordering.non_path_children_dso

    ### Validate path definitions
    for ii in range(o.natoms):
        # The subpath child must:
        #   not exist (the node is a leaf)
        #   point back to the node as its parent
        first_child = o.subpath_child_ko[ii]
        assert first_child == -1 or o.parent_ko[first_child] == ii

        # Each non-subpath child must:
        #   be non-existant (the node has < max num non-subpath children)
        #   point back to the node as its parent
        for jj in range(o.non_path_children_ko.shape[1]):
            child = o.non_path_children_ko[ii, jj]
            assert child == -1 or o.parent_ko[child] == ii

    ### Validate refold ordering
    ro: RefoldOrdering = o.refold_ordering
    for ii_ki in range(o.natoms):
        parent_ki = kintree.parent[ii_ki]

        # The node's parent must:
        #   be "self" (as the node is root?)
        #   be off the subpath (as the node is a subpath root?)
        #   be the kinematic parent (as the node is on a subpath?)
        ii_ri = ro.ki2ri[ii_ki]
        parent_ri = ro.ki2ri[parent_ki]
        assert parent_ki == ii_ki or \
            ro.non_subpath_parent[ii_ri] == -1 or \
            ro.non_subpath_parent[ii_ri] == parent_ri

        # The node's child must:
        #    be non-existant (as the node is a leaf?)
        #    not have a non-subpath parent (as it's parent is this node?)
        child_ki = o.subpath_child_ko[ii_ki]
        assert (
            child_ki == -1 or ro.non_subpath_parent[ro.ki2ri[child_ki]] == -1
        )

    ### Validate derivsum ordering
    do: DerivsumOrdering = o.derivsum_ordering
    for ii in range(o.natoms):
        # Each non-subpath child must:
        #   be non-existant (the node has < max num non-subpath children)
        #   point back to the node as its parent
        for jj in range(do.non_path_children.shape[1]):
            child = do.non_path_children[ii, jj]
            ii_ki = do.dsi2ki[ii]
            assert child == -1 or ii_ki == o.parent_ko[do.dsi2ki[child]]
