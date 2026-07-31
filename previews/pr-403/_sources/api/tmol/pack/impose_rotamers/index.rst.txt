tmol.pack.impose_rotamers
=========================

.. py:module:: tmol.pack.impose_rotamers


Functions
---------

.. autoapisummary::

   tmol.pack.impose_rotamers.impose_top_rotamer_assignments


Module Contents
---------------

.. py:function:: impose_top_rotamer_assignments(orig_pose_stack: tmol.pose.pose_stack.PoseStack, rotamer_set: tmol.pack.rotamer.build_rotamers.RotamerSet, rotamer_for_nonmolten_block: tmol.types.torch.Tensor[torch.int64][:, :], n_molten_blocks_per_pose: tmol.types.torch.Tensor[torch.int64][:], bc_rot_offset_for_molten_block: tmol.types.torch.Tensor[torch.int64][:, :], bc_rot_to_orig_rot: tmol.types.torch.Tensor[torch.int64][:], bc_assignment: tmol.types.torch.Tensor[torch.int32][:, :, :])

   .. rubric:: Docstring

   .. code-block:: text

      Impose the lowest-energy rotamer assignemnt to each pose in the original PoseStack.
      

