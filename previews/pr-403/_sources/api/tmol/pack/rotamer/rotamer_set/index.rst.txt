tmol.pack.rotamer.rotamer_set
=============================

.. py:module:: tmol.pack.rotamer.rotamer_set


Classes
-------

.. autoapisummary::

   tmol.pack.rotamer.rotamer_set.RotamerSet


Module Contents
---------------

.. py:class:: RotamerSet

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: n_rots_for_pose
      :type:  tmol.types.torch.Tensor[torch.int64][:]


   .. py:attribute:: rot_offset_for_pose
      :type:  tmol.types.torch.Tensor[torch.int64][:]


   .. py:attribute:: n_rots_for_block
      :type:  tmol.types.torch.Tensor[torch.int64][:, :]


   .. py:attribute:: rot_offset_for_block
      :type:  tmol.types.torch.Tensor[torch.int64][:, :]


   .. py:attribute:: pose_for_rot
      :type:  tmol.types.torch.Tensor[torch.int64][:]


   .. py:attribute:: block_type_ind_for_rot
      :type:  tmol.types.torch.Tensor[torch.int64][:]


   .. py:attribute:: block_ind_for_rot
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: coord_offset_for_rot
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: coords
      :type:  tmol.types.torch.Tensor[torch.float32][:, 3]


   .. py:attribute:: first_rot_block_type
      :type:  tmol.types.torch.Tensor[torch.int64][:, :]


   .. py:attribute:: max_n_rots_per_pose
      :type:  int


   .. py:attribute:: pose_ind_for_atom
      :type:  tmol.types.torch.Tensor[torch.int64][:]


   .. py:property:: n_rotamers_total


