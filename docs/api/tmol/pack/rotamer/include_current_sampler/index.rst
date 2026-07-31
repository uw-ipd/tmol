tmol.pack.rotamer.include_current_sampler
=========================================

.. py:module:: tmol.pack.rotamer.include_current_sampler


Classes
-------

.. autoapisummary::

   tmol.pack.rotamer.include_current_sampler.IncludeCurrentSampler


Functions
---------

.. autoapisummary::

   tmol.pack.rotamer.include_current_sampler.create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler


Module Contents
---------------

.. py:class:: IncludeCurrentSampler

   Bases: :py:obj:`tmol.pack.rotamer.conformer_sampler.ConformerSampler`


   .. py:method:: sampler_name()
      :classmethod:



   .. py:method:: annotate_residue_type(rt: tmol.chemical.restypes.RefinedResidueType)


   .. py:method:: annotate_packed_block_types(packed_block_types: tmol.pose.packed_block_types.PackedBlockTypes)


   .. py:method:: defines_rotamers_for_rt(rt: tmol.chemical.restypes.RefinedResidueType)


   .. py:method:: defines_rotamers_for_bts(pbt: tmol.pose.packed_block_types.PackedBlockTypes, bt_inds: tmol.types.torch.Tensor[torch.int64]) -> tmol.types.torch.Tensor[torch.bool]


   .. py:method:: first_sc_atoms_for_rt(rt: tmol.chemical.restypes.RefinedResidueType) -> Tuple[str, Ellipsis]


   .. py:method:: create_samples_for_poses(pose_stack: tmol.pose.pose_stack.PoseStack, task: SetPackerTask) -> Tuple[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int32][:], dict]


   .. py:method:: fill_dofs_for_samples(pose_stack: tmol.pose.pose_stack.PoseStack, task: PackerTask, orig_kinforest: tmol.kinematics.datatypes.KinForest, orig_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9], gbt_for_conformer: tmol.types.torch.Tensor[torch.int64][:], block_type_ind_for_conformer: tmol.types.torch.Tensor[torch.int64][:], n_dof_atoms_offset_for_conformer: tmol.types.torch.Tensor[torch.int64][:], conformer_built_by_sampler: tmol.types.torch.Tensor[torch.bool][:], conf_inds_for_sampler: tmol.types.torch.Tensor[torch.int64][:], sampler_n_rots_for_gbt: tmol.types.torch.Tensor[torch.int32][:], sampler_gbt_for_rotamer: tmol.types.torch.Tensor[torch.int32][:], sample_dict: dict, conf_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9])


.. py:function:: create_full_dof_inds_to_copy_from_orig_to_rotamers_for_include_current_sampler(poses: tmol.pose.pose_stack.PoseStack, task: SetPackerTask, gbt_for_rot: tmol.types.torch.Tensor[torch.int64][:], block_type_ind_for_rot: tmol.types.torch.Tensor[torch.int64][:], conf_inds_for_sampler: tmol.types.torch.Tensor[torch.int64][:], sampler_n_rots_for_gbt: tmol.types.torch.Tensor[torch.int32][:], sampler_gbt_for_rotamer: tmol.types.torch.Tensor[torch.int32][:], n_dof_atoms_offset_for_rot: tmol.types.torch.Tensor[torch.int64][:]) -> Tuple[tmol.types.torch.Tensor[torch.int64][:], tmol.types.torch.Tensor[torch.int64][:]]

