import pytest
import torch

from tmol.kinematics.builder import KinematicBuilder

from tmol.kinematics.gpu_operations.scan_paths import GPUKinTreeReordering


def test_gpu_refold_data_construction(ubq_system, torch_device):
    tsys = ubq_system
    kintree = KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, tsys.bonds)
    ).kintree.to(device=torch_device)

    ### If kintree is cpu resident then the gpu parallel scan is invalid
    if torch_device.type == "cpu":
        with pytest.raises(ValueError):
            GPUKinTreeReordering.calculate_from_kintree(kintree)
        return

    ### Otherwise test the derived ordering
    ordering = GPUKinTreeReordering.calculate_from_kintree(
        kintree, torch.device("cuda")
    )

    # Extract path data from tree reordering.
    natoms = ordering.natoms
    subpath_child_ko = ordering.subpath_child_ko
    ki2ri = ordering.ki2ri.copy_to_host()
    dsi2ki = ordering.dsi2ki
    parent_ko = kintree.parent
    non_subpath_parent_ro = ordering.non_subpath_parent_ro.copy_to_host()
    subpath_child_ko = ordering.subpath_child_ko
    non_path_children_ko = ordering.non_path_children_ko
    non_path_children_dso = ordering.non_path_children_dso.copy_to_host()

    for ii_ki in range(natoms):
        parent_ki = kintree.parent[ii_ki]

        ii_ri = ki2ri[ii_ki]
        parent_ri = ki2ri[parent_ki]
        assert parent_ki == ii_ki or \
            non_subpath_parent_ro[ii_ri] == -1 or \
            non_subpath_parent_ro[ii_ri] == parent_ri

        child_ki = subpath_child_ko[ii_ki]
        assert child_ki == -1 or non_subpath_parent_ro[ki2ri[child_ki]] == -1

    for ii in range(natoms):
        for jj in range(non_path_children_ko.shape[1]):
            child = non_path_children_ko[ii, jj]
            assert child == -1 or parent_ko[child] == ii
        first_child = subpath_child_ko[ii]
        assert first_child == -1 or parent_ko[first_child] == ii

    for ii in range(natoms):
        for jj in range(non_path_children_ko.shape[1]):
            child = non_path_children_dso[ii, jj]
            ii_ki = dsi2ki[ii]
            assert child == -1 or ii_ki == parent_ko[dsi2ki[child]]
