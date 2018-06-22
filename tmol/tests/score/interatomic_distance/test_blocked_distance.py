# import torch

# from tmol.score.interatomic_distance import BlockedDistanceAnalysis

# def test_blocked_interatomic_distance_nulls(multilayer_test_coords):
#     """Test that interatomic distance properly handles null blocks."""
#     null_padded = multilayer_test_coords.new_full((4, 24 + 8, 3), nan)
#     null_padded[:, :24, :] = multilayer_test_coords

#     bda = BlockedDistanceAnalysis.setup(
#         block_size=8, coords=multilayer_test_coords
#     )
#     np_bda = BlockedDistanceAnalysis.setup(block_size=8, coords=null_padded)
#     assert (bda.block_triu_inds == np_bda.block_triu_inds).all()

# def test_blocked_interatomic_distance_layered(multilayer_test_coords):
#     threshold_distance = 6.0

#     # dense_expected_block_interactions = torch.Tensor([
#     #     [  #0-------1------2
#     #         [1, 0, 0],
#     #         [0, 1, 0],
#     #         [0, 0, 1],
#     #     ],
#     #     [  #2-------1------0
#     #         [1, 0, 0],
#     #         [0, 1, 0],
#     #         [0, 0, 1],
#     #     ],
#     #     [  #----0---1---2---
#     #         [1, 1, 0],
#     #         [1, 1, 1],
#     #         [0, 1, 1],
#     #     ],
#     #     [  #-------012------
#     #         [1, 1, 1],
#     #         [1, 1, 1],
#     #         [1, 1, 1],
#     #     ],
#     # ]).to(dtype=torch.uint8)

#     triu_expected_block_interactions = torch.Tensor([
#         [2, 0],
#         [2, 2],
#         [3, 0],
#         [3, 1],
#         [3, 2],
#     ]).to(dtype=torch.long)

#     bda = BlockedDistanceAnalysis.setup(
#         coords=multilayer_test_coords,
#         block_size=8,
#     )
#     assert (
#         bda.block_triu_inds ==
#         torch.Tensor([[0, 1], [0, 2], [1, 2]]).to(dtype=torch.long)
#     ).all() # yapf: disable
#     assert (
#         (bda.block_triu_min_dist < threshold_distance).nonzero() ==
#         triu_expected_block_interactions
#     # ).all() # yapf: disable
