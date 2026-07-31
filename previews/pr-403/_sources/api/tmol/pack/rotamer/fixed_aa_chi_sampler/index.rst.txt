tmol.pack.rotamer.fixed_aa_chi_sampler
======================================

.. py:module:: tmol.pack.rotamer.fixed_aa_chi_sampler


Classes
-------

.. autoapisummary::

   tmol.pack.rotamer.fixed_aa_chi_sampler.FixedAAChiSampler


Module Contents
---------------

.. py:class:: FixedAAChiSampler

   Bases: :py:obj:`tmol.pack.rotamer.chi_sampler.ChiSampler`


   .. py:method:: sampler_name()
      :classmethod:



   .. py:method:: defines_rotamers_for_rt(rt: tmol.chemical.restypes.RefinedResidueType)


   .. py:method:: defines_rotamers_for_bts(pbt: tmol.pose.packed_block_types.PackedBlockTypes, bt_inds: tmol.types.torch.Tensor[torch.int64]) -> tmol.types.torch.Tensor[torch.bool]


   .. py:method:: first_sc_atoms_for_rt(rt: tmol.chemical.restypes.RefinedResidueType) -> Tuple[str, Ellipsis]


   .. py:method:: annotate_residue_type(block_type)


   .. py:method:: annotate_packed_block_types(packed_block_types)


   .. py:method:: sample_chi_for_poses(poses: tmol.pose.pose_stack.PoseStack, task: tmol.pack.packer_task.SetPackerTask) -> Tuple[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.float32][:, :]]


