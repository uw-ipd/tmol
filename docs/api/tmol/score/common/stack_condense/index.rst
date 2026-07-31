tmol.score.common.stack_condense
================================

.. py:module:: tmol.score.common.stack_condense


Functions
---------

.. autoapisummary::

   tmol.score.common.stack_condense.condense_numpy_inds
   tmol.score.common.stack_condense.condense_torch_inds
   tmol.score.common.stack_condense.take_values_w_sentineled_index
   tmol.score.common.stack_condense.take_values_w_sentineled_index_and_dest
   tmol.score.common.stack_condense.take_values_w_sentineled_dest
   tmol.score.common.stack_condense.condense_subset
   tmol.score.common.stack_condense.take_condensed_3d_subset
   tmol.score.common.stack_condense.tile_subset_indices
   tmol.score.common.stack_condense.arg_tile_subset_indices


Module Contents
---------------

.. py:function:: condense_numpy_inds(selection: tmol.types.array.NDArray[bool][:, :])

   .. rubric:: Docstring

   .. code-block:: text

      Given a two dimensional boolean tensor, create
      an output tensor holding the column indices of the non-zero
      entries for each row. Pad out the extra entries
      in any given row that do not correspond to a selected
      entry with a sentinel of -1.
      
      e.g. if the input is
      
      [[ 0  1  0  1]
      [  1  1  0  1]]
      
      then the output will be
      
      [[ 1  3 -1]
      [  0  1  3]]
      

.. py:function:: condense_torch_inds(selection: tmol.types.torch.Tensor[bool][:, :], device: torch.device)

   .. rubric:: Docstring

   .. code-block:: text

      Given a two dimensional boolean tensor, create
      an output tensor holding the column indices of the non-zero
      entries for each row. Pad out the extra entries
      in any given row that do not correspond to a selected
      entry with a sentinel of -1.
      
      e.g. if the input is
      [[ 0  1  0  1]
      [  1  1  0  1]]
      then the output will be
      [[ 1  3 -1]
      [  0  1  3]]
      

.. py:function:: take_values_w_sentineled_index(value_tensor, sentineled_index_tensor: tmol.types.torch.Tensor[torch.int64][:, :], default_fill=-1)

   .. rubric:: Docstring

   .. code-block:: text

      The sentinel in the sentineled_index_tensor is -1: the positions
      with the sentinel value should not be used as an index into the
      value tensor. This function returns a tensor of the same shape as
      the sentineled_index_tensor with a dtype of the value tensor.
      
      E.g. if the value tensor is [10 11 12 13 14 15]
      and the sentineled_index_tensor is
      
      [[ 2 1 2 5 -1]
      [  1 4 1 5  2]]
      
      then the output tensor will be
      
      [[ 12 11 12 15 -1]
      [  11 14 11 15  12]]
      

.. py:function:: take_values_w_sentineled_index_and_dest(value_tensor, sentineled_index_tensor: tmol.types.torch.Tensor[torch.int64][:, :], sentineled_dest_tensor, default_fill=-1)

   .. rubric:: Docstring

   .. code-block:: text

      The sentinel in the sentineled_index_tensor is -1: the positions
      with the sentinel value should not be used as an index into the
      value tensor. The sentinel in the sentineled_dest_tensor is also
      -1: the positions with the sentinel value should not be written
      to in the output tensor. This function returns a tensor of the
      same shape as the sentineled_dest_tensor with a dtype of the
      value tensor, which is indexed into using the
      sentineled_index_tensor. The values in the sentineled_dest_tensor
      do not matter except where they are -1.
      
      E.g. if the value tensor is [10 11 12 13 14 15],
      the sentineled_index_tensor is
      [[ 2 -1  2  5 -1]
      [  1  4 -1  5  2]],
      and the sentineled_dest_tensor is
      [[ 1  1  1 -1]
      [  1  1  1  1]]
      
      then the output tensor will be
      [[ 12 12 15 -1]
      [  11 14 15 12]]
      

.. py:function:: take_values_w_sentineled_dest(value_tensor, values_to_take, sentineled_dest_tensor, default_fill=-1)

   .. rubric:: Docstring

   .. code-block:: text

      Take a subset of the values from the value_tensor indicated by
      the boolean values_to_take tensor, and write them into an output
      tensor in a shape with non-negative-one values in the
      sentineled_dest_tensor. There need to be as many "true" values in
      the values_to_take tensor as they are non-negative-one values
      in the sentineled_dest_tensor.
      
      E.g. if the value tensor is
      [[10 11 12 13 14],
      [ 20 21 22 23 24]]
      the values_to_take tensor is
      [[ 1  0  1  1  0]
      [  1  1  0  1  1]],
      and the sentineled_dest_tensor is
      [[ 1  1  1 -1]
      [  1  1  1  1]]
      
      then the output tensor will be
      [[10 12 13 -1]
      [ 20 21 23 24]]
      

.. py:function:: condense_subset(values, values_to_keep, default_fill=-1)

   .. rubric:: Docstring

   .. code-block:: text

      Take the values for the third dimension of the 3D "values" tensor,
      (condensing them), corresponding to the positions indicated by
      the values_to_keep tensor.
      
      E.g. if the values tensor is
      [[[10 10] [11 11] [12 12] [13 13] [14 14]],
      [ [20 20] [21 21] [22 22] [23 23] [24 24]]]
      the values_to_keep tensor is
      [[1 0 1 1 0]
      [ 1 1 0 1 1]]
      
      then the output tensor will be
      [[ [10 10] [12 12] [13 13] [ -1 -1]]
      [  [20 20] [21 21] [23 23] [24 24]]]
      

.. py:function:: take_condensed_3d_subset(values, condensed_inds_to_keep: tmol.types.torch.Tensor[torch.int64][:, :], condensed_dst_inds: tmol.types.torch.Tensor[torch.int64][:, 2], default_fill=-1)

   .. rubric:: Docstring

   .. code-block:: text

      Take the values for the third dimension of the 3D "values" tensor,
      at the positions indicated by the "condensed_inds_to_keep" tensor,
      and writing them to the indices indicated by the "condensed_dst_inds".
      This function is equivalent to the above "condense_subset" function
      if that function's "values_to_keep" tensor is converted to the
      inputs to this function with the following operations:
      
      condensed_inds_to_keep = condense_torch_inds(values_to_keep != -1, device)
      condensed_dst_inds = torch.nonzero(inds_to_keep != -1)
      
      This function is more efficient if you intend to use the
      "condensed_inds_to_keep" or the "condensed_dst_inds" tensors multiple
      times.
      
      E.g. if the values tensor is
      [[[10 10] [11 11] [12 12] [13 13] [14 14]],
      [ [20 20] [21 21] [22 22] [23 23] [24 24]]]
      the condensed_inds_to_keep tensor is
      [[ 0 -1  2  3]
      [  4  3  2  4]],
      and the condensed_dest_tensor is
      [[ 0 0]
      [  0 1]
      [  0 2]
      [  1 0]
      [  1 1]
      [  1 2]
      [  1 3]]
      
      then the output tensor will be
      [[ [10 10] [12 12] [13 13] [ -1 -1]]
      [  [24 24] [23 23] [22 22] [24 24]]]
      

.. py:function:: tile_subset_indices(indices: Union[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int64][:], tmol.types.array.NDArray[numpy.int32][:], tmol.types.array.NDArray[numpy.int64][:]], tile_size: int, max_entry: Optional[int] = None)

   .. rubric:: Docstring

   .. code-block:: text

      Take the indices of a subset of things and "tile" them so that they're
      in groups based on the equivalence class `i // tile_size` and left-justify
      the indices within the tile.
      
      E.g.
      If the subset indices are [0, 3, 4, 7, 10, 12, 14]
      and the tile_size is 8,
      then the output will be:
      [0, 3, 4, 7, -1, -1, -1, -1, 2, 4, 6, -1, -1, -1, -1, -1] and
      [4, 3]
      representing the tiling of the indices and the number of indices per tile,
      reflecting there being two tiles where there are four values in
      the first tile and three values in the second tile.
      The indices are given as tile indices so that 10-->2,
      12-->4, 14-->6. The entries that are in the first tile remain
      unchanged, of course.
      
      If desired, a maximum index can be given so that a desired number
      of tiles can be created even if the subset includes no entries for
      the last tile.
      
      E.g.
      If the subset indices are [0, 3, 4, 7]
      and the tile size is 8 and the max_entry is 15, then two tiles are desired
      and the output will be:
      [0, 3, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] and
      [4, 0]
      representing the tiling of the indices and the number of indices per tile
      
      Works for both torch and numpy inputs.
      

.. py:function:: arg_tile_subset_indices(indices: Union[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int64][:], tmol.types.array.NDArray[numpy.int32][:], tmol.types.array.NDArray[numpy.int64][:]], tile_size: int, max_entry: Optional[int] = None)

   .. rubric:: Docstring

   .. code-block:: text

      Take the indices of a subset of things and return the indices (args) that
      would "tile" them so that they're in groups based on the equivalence class
      `i // tile_size` and left-justify those indices within the tiles.
      Having the indices of the tiled subset is desired in cases when there
      is additional data for the subset that also needs to be tiled.
      
      E.g.
      If the subset indices are [0, 3, 4, 7, 10, 12, 14]
      and the tile_size is 8,
      then the output will be:
      [0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, -1, -1, -1, -1, -1] and
      [4, 3]
      representing the tiling of the indices by their indices in the input array
      (confusingly named) indices (!) and the number of indices per tile,
      reflecting there being two tiles, where there are four values in the
      first tile and three values in the second tile.
      
      If desired, a maximum index can be given so that a desired number
      of tiles can be created even if the subset includes no entries for
      the last tile.
      
      E.g.
      If the subset indices are [0, 3, 4, 7]
      and the tile size is 8 and the max_entry is 15, then two tiles are desired
      and the output will be:
      [0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1] and
      [4, 0]
      representing the tiling of the indices by their indices and the number
      of indices per tile.
      
      Works for both torch and numpy inputs.
      

