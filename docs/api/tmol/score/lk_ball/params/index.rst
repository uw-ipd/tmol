tmol.score.lk_ball.params
=========================

.. py:module:: tmol.score.lk_ball.params


Classes
-------

.. autoapisummary::

   tmol.score.lk_ball.params.LKBallBlockTypeParams
   tmol.score.lk_ball.params.LKBallPackedBlockTypesParams


Module Contents
---------------

.. py:class:: LKBallBlockTypeParams

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: tile_n_polar_atoms
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: tile_n_occluder_atoms
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: tile_pol_occ_inds
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: tile_lk_ball_params
      :type:  tmol.types.array.NDArray[numpy.float32][:, :, 9]


.. py:class:: LKBallPackedBlockTypesParams

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: tile_n_polar_atoms
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: tile_n_occluder_atoms
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: tile_pol_occ_inds
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: tile_lk_ball_params
      :type:  tmol.types.torch.Tensor[torch.float32][:, :, :, 9]


