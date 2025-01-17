# import pytest

# import torch
# import numpy

# from tmol.kinematics.operations import inverseKin, forwardKin
# from tmol.kinematics.builder import _KinematicBuilder
# from tmol.kinematics.scan_ordering import KinForestScanOrdering
# from tmol.tests.torch import requires_cuda
# from tmol.io import pose_stack_from_pdb
# from tmol.pose.pose_stack import PoseStack
# from tmol.kinematics.fold_forest import FoldForest
# from tmol.kinematics.scan_ordering import construct_kin_module_data_for_pose


# def system_kinforest(target_system):
#     tsys = target_system
#     roots = numpy.zeros((1,), dtype=numpy.int32)
#     return (
#         _KinematicBuilder()
#         .append_connected_components(
#             roots,
#             *_KinematicBuilder.bonds_to_forest(roots, tsys.bonds.astype(numpy.int32)),
#         )
#         .kinforest
#     )

# def kinforest_for_pdb(pdb, device):
#     pose_stack = pose_stack_from_pdb(pdb, device)
#     ff = FoldForest.polymeric_forest(numpy.array([pose_stack.max_n_blocks], dtype=numpy.int32))
#     kmd = construct_kin_module_data_for_pose(pose_stack, torch.tensor(ff.edges, dtype=torch.int32))
#     return pose_stack, kmd.forest


# # @pytest.mark.benchmark(group="score_setup")
# # def test_refold_data_construction(benchmark, torch_device, ubq_pdb):
# #     # kinforest = system_kinforest(ubq_system)
# #     _, kinforest = kinforest_for_pdb(ubq_pdb, torch_device)

# #     @benchmark
# #     def tree_reordering_cpp() -> KinForestScanOrdering:
# #         return KinForestScanOrdering.calculate_from_kinforest(kinforest)

# #     kinorder = tree_reordering_cpp

# #     # ensure the tree is reasonable:
# #     #   - dimensionality match
# #     #   - all connected nodes are parent->child
# #     natms = kinforest.id.shape[0]
# #     generated = numpy.zeros(natms, dtype=numpy.int32)
# #     ngens = len(kinorder.forward_scan_paths.gens)

# #     for i in range(ngens - 1):
# #         scanstart = kinorder.forward_scan_paths.gens[i][1]
# #         scanstop = kinorder.forward_scan_paths.gens[i + 1][1]
# #         for j in range(scanstart, scanstop):
# #             nodestart = (
# #                 kinorder.forward_scan_paths.gens[i][0]
# #                 + kinorder.forward_scan_paths.scans[j]
# #             )
# #             nodestop = kinorder.forward_scan_paths.gens[i + 1][0]
# #             if j < scanstop - 1:
# #                 nodestop = (
# #                     kinorder.forward_scan_paths.gens[i][0]
# #                     + kinorder.forward_scan_paths.scans[j + 1]
# #                 )

# #             # tag the root(s) as generated
# #             if i == 0:
# #                 generated[kinorder.forward_scan_paths.nodes[nodestart]] += 1

# #             for k in range(nodestart, nodestop - 1):
# #                 parent = kinorder.forward_scan_paths.nodes[k]
# #                 child = kinorder.forward_scan_paths.nodes[k + 1]
# #                 assert kinforest.parent[child].to(dtype=torch.int) == parent

# #                 # tag the child as visited
# #                 generated[child] += 1

# #     #  ensure all nodes are generated exactly once
# #     for i in range(1, natms):
# #         assert generated[i] == 1


# @requires_cuda
# @pytest.mark.benchmark(group="kinematic_op_micro_forward")
# def test_refold_values_cpp(benchmark, big_pdb):
#     target_device = torch.device("cuda")
#     # kinforest = system_kinforest(big_system)
#     pose_stack, kinforest = kinforest_for_pdb(big_pdb, target_device)

#     tcoords = torch.tensor(pose_stack.coords.view(-1, 3)[kinforest.id]).to(dtype=torch.float64, device=target_device)
#     tkinforest = kinforest.to(device=target_device)
#     bkin = inverseKin(tkinforest, tcoords)

#     # KinForestScanOrdering.calculate_from_kinforest(tkinforest)

#     @benchmark
#     def parallel_refold_hts_cpp():
#         return forwardKin(tkinforest, bkin)

#     # fold via cpu and gpu, ensuring results match
#     dofs_cuda = parallel_refold_hts_cpp

#     bkin = inverseKin(kinforest, tcoords.cpu())
#     dofs_cpu = forwardKin(kinforest, bkin)

#     assert dofs_cuda.device.type == "cuda"
#     assert dofs_cpu.device.type == "cpu"
#     torch.testing.assert_close(dofs_cuda.cpu(), dofs_cpu)


# @requires_cuda
# @pytest.mark.benchmark(group="kinematic_op_micro_backward")
# def test_derivsum_values_cpp(benchmark, big_system):
#     target_device = torch.device("cuda")
#     torch.manual_seed(1663)

#     kinforest_cpu = system_kinforest(big_system)
#     coords_cpu = torch.tensor(big_system.coords[kinforest_cpu.id])
#     dscdx_cpu = (torch.rand_like(coords_cpu) * 0.2) - 0.1

#     coords_cuda = coords_cpu.to(device=target_device)
#     kinforest_cuda = kinforest_cpu.to(device=target_device)
#     dscdx_cuda = dscdx_cpu.to(device=target_device)

#     bkin_cpu = inverseKin(kinforest_cpu, coords_cpu, requires_grad=True)
#     recoords_cpu = forwardKin(kinforest_cpu, bkin_cpu)

#     bkin_cuda = inverseKin(kinforest_cuda, coords_cuda, requires_grad=True)
#     recoords_cuda = forwardKin(kinforest_cuda, bkin_cuda)

#     @benchmark
#     def parallel_derivsum_cuda():
#         return torch.autograd.grad(
#             recoords_cuda,
#             bkin_cuda.raw,
#             dscdx_cuda,
#             retain_graph=True,
#             allow_unused=True,
#         )

#     (dscddof_cuda,) = parallel_derivsum_cuda

#     # same calc on CPU
#     (dscddof_cpu,) = torch.autograd.grad(
#         recoords_cpu, bkin_cpu.raw, dscdx_cpu, retain_graph=True, allow_unused=True
#     )

#     assert dscddof_cuda.device.type == "cuda"
#     assert dscddof_cpu.device.type == "cpu"

#     # angle between vectors should be close to 0
#     norm_a = torch.sqrt(torch.sum(dscddof_cuda.cpu() * dscddof_cuda.cpu()))
#     norm_b = torch.sqrt(torch.sum(dscddof_cpu * dscddof_cpu))

#     # with the scan bugfix, these two tensors are so close that numerical noise
#     # when taking their dot product results in a number > 1 and the arccos then
#     # ends up as NaN

#     # angle = torch.acos(
#     #         torch.sum(dscddof_cuda.cpu() * dscddof_cpu) / (norm_a * norm_b)
#     #     )
#     # assert torch.abs(angle) < 1e-2
#     numpy.testing.assert_almost_equal(
#         1.0, (torch.sum(dscddof_cuda.cpu() * dscddof_cpu) / (norm_a * norm_b)).numpy()
#     )
