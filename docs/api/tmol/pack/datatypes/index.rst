tmol.pack.datatypes
===================

.. py:module:: tmol.pack.datatypes


Classes
-------

.. autoapisummary::

   tmol.pack.datatypes.PackerEnergyTables


Module Contents
---------------

.. py:class:: PackerEnergyTables

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. py:attribute:: max_n_rotamers_per_pose
      :type:  int


   .. py:attribute:: pose_n_res
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: pose_n_rotamers
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: pose_rotamer_offset
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: nrotamers_for_res
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: oneb_offsets
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: res_for_rot
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: chunk_size
      :type:  int


   .. py:attribute:: chunk_offset_offsets
      :type:  tmol.types.torch.Tensor[torch.int64][:, :, :]


   .. py:attribute:: chunk_offsets
      :type:  tmol.types.torch.Tensor[torch.int64][:]


   .. py:attribute:: energy1b
      :type:  tmol.types.torch.Tensor[torch.float32][:]


   .. py:attribute:: energy2b
      :type:  tmol.types.torch.Tensor[torch.float32][:]


