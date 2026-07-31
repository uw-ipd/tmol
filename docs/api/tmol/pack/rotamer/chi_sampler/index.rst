tmol.pack.rotamer.chi_sampler
=============================

.. py:module:: tmol.pack.rotamer.chi_sampler


Classes
-------

.. autoapisummary::

   tmol.pack.rotamer.chi_sampler.ChiSampler


Functions
---------

.. autoapisummary::

   tmol.pack.rotamer.chi_sampler.copy_dofs_from_orig_to_rotamers_for_sampler
   tmol.pack.rotamer.chi_sampler.create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler
   tmol.pack.rotamer.chi_sampler.assign_chi_dofs_from_samples


Module Contents
---------------

.. py:class:: ChiSampler

   Bases: :py:obj:`tmol.pack.rotamer.conformer_sampler.ConformerSampler`


   .. py:method:: sampler_name()
      :classmethod:

      :abstractmethod:



   .. py:method:: annotate_residue_type(rt: tmol.chemical.restypes.RefinedResidueType)


   .. py:method:: annotate_packed_block_types(packed_block_types: tmol.pose.packed_block_types.PackedBlockTypes)


   .. py:method:: defines_rotamers_for_rt(rt: tmol.chemical.restypes.RefinedResidueType)
      :abstractmethod:



   .. py:method:: first_sc_atoms_for_rt(rt_name: str) -> Tuple[str, Ellipsis]
      :abstractmethod:



   .. py:method:: create_samples_for_poses(pose_stack: tmol.pose.pose_stack.PoseStack, task: PackerTask) -> Tuple[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int32][:], dict]


   .. py:method:: sample_chi_for_poses(systems: tmol.pose.pose_stack.PoseStack, task: PackerTask) -> Tuple[tmol.types.torch.Tensor[torch.int32][:, :, :], tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.float32][:, :]]
      :abstractmethod:



   .. py:method:: fill_dofs_for_samples(pose_stack: tmol.pose.pose_stack.PoseStack, task: PackerTask, orig_kinforest: tmol.kinematics.datatypes.KinForest, orig_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9], gbt_for_conformer: tmol.types.torch.Tensor[torch.int64][:], block_type_ind_for_conformer: tmol.types.torch.Tensor[torch.int64][:], n_dof_atoms_offset_for_conformer: tmol.types.torch.Tensor[torch.int64][:], conformer_built_by_sampler: tmol.types.torch.Tensor[torch.bool][:], conf_inds_for_sampler: tmol.types.torch.Tensor[torch.int64][:], sampler_n_rots_for_gbt: tmol.types.torch.Tensor[torch.int32][:], sampler_gbt_for_rotamer: tmol.types.torch.Tensor[torch.int32][:], sample_dict: dict, conf_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9])


.. py:function:: copy_dofs_from_orig_to_rotamers_for_sampler(poses: tmol.pose.pose_stack.PoseStack, task, sampler_name: str, gbt_for_rot: tmol.types.torch.Tensor[torch.int64][:], block_type_ind_for_rot: tmol.types.torch.Tensor[torch.int64][:], conf_inds_for_sampler: tmol.types.torch.Tensor[torch.int64][:], sampler_n_rots_for_gbt: tmol.types.torch.Tensor[torch.int32][:], sampler_gbt_for_rotamer: tmol.types.torch.Tensor[torch.int32][:], n_dof_atoms_offset_for_rot: tmol.types.torch.Tensor[torch.int64][:], orig_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9], rot_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9])

.. py:function:: create_dof_inds_to_copy_from_orig_to_rotamers_for_sampler(poses: tmol.pose.pose_stack.PoseStack, task: PackerTask, sampler_name: str, gbt_for_rot: tmol.types.torch.Tensor[torch.int64][:], block_type_ind_for_rot: tmol.types.torch.Tensor[torch.int64][:], conf_inds_for_sampler: tmol.types.torch.Tensor[torch.int64][:], sampler_n_rots_for_gbt: tmol.types.torch.Tensor[torch.int32][:], sampler_gbt_for_rotamer: tmol.types.torch.Tensor[torch.int32][:], n_dof_atoms_offset_for_rot: tmol.types.torch.Tensor[torch.int64][:]) -> Tuple[tmol.types.torch.Tensor[torch.int64][:], tmol.types.torch.Tensor[torch.int64][:]]

.. py:function:: assign_chi_dofs_from_samples(pbt: tmol.pose.packed_block_types.PackedBlockTypes, block_type_ind_for_rot: tmol.types.torch.Tensor[torch.int64][:], conf_inds_for_sampler: tmol.types.torch.Tensor[torch.int64][:], sampler_n_rots_for_bt: tmol.types.torch.Tensor[torch.int32][:], sampler_gbt_for_rotamer: tmol.types.torch.Tensor[torch.int32][:], n_dof_atoms_offset_for_rot: tmol.types.torch.Tensor[torch.int64][:], chi_atoms: tmol.types.torch.Tensor[torch.int32][:, :], chi: tmol.types.torch.Tensor[torch.float32][:, :], rot_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9])

