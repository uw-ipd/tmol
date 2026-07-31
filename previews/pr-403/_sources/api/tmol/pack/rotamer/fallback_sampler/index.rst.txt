tmol.pack.rotamer.fallback_sampler
==================================

.. py:module:: tmol.pack.rotamer.fallback_sampler


Classes
-------

.. autoapisummary::

   tmol.pack.rotamer.fallback_sampler.FallbackSampler


Module Contents
---------------

.. py:class:: FallbackSampler

   Bases: :py:obj:`tmol.pack.rotamer.conformer_sampler.ConformerSampler`


   .. rubric:: Docstring

   .. code-block:: text

      Include the input conformation as a rotamer only for positions that have
      no rotamers from any other sampler.
      
      This is the default sampler in PackerPalette. Unlike IncludeCurrentSampler,
      it does not unconditionally add a rotamer for every position; instead it
      activates only when every other sampler in the block-level task returns
      False from defines_rotamers_for_rt for the original block type, ensuring
      that positions covered by, e.g., DunbrackChiSampler do not accumulate an
      extra current-conformation rotamer.
      
      The disable_packing case (all block types disallowed) is also handled: a
      rotamer from the input conformation is always produced so the packer has
      something to represent for fixed residues.
      

   .. py:method:: sampler_name()
      :classmethod:



   .. py:method:: annotate_residue_type(rt: tmol.chemical.restypes.RefinedResidueType)


   .. py:method:: annotate_packed_block_types(packed_block_types: tmol.pose.packed_block_types.PackedBlockTypes)


   .. py:method:: defines_rotamers_for_rt(rt: tmol.chemical.restypes.RefinedResidueType)


   .. py:method:: first_sc_atoms_for_rt(rt: tmol.chemical.restypes.RefinedResidueType) -> Tuple[str, Ellipsis]


   .. py:method:: create_samples_for_poses(pose_stack: tmol.pose.pose_stack.PoseStack, task: SetPackerTask) -> Tuple[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int32][:], dict]

      .. rubric:: Docstring

      .. code-block:: text

         Create rotamers for the blocks that either (1) have no allowed block types, in which case the residue
         is considered fixed and we simply create a rotamer of the input conformation, or (2) have no conformer
         sampler that defines conformers for it. So the first step is to look at the other conformers stored
         in the SetPackerTask and ask them which block types they define rotamers for.
         


   .. py:method:: fill_dofs_for_samples(pose_stack: tmol.pose.pose_stack.PoseStack, task: SetPackerTask, orig_kinforest: tmol.kinematics.datatypes.KinForest, orig_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9], gbt_for_conformer: tmol.types.torch.Tensor[torch.int64][:], block_type_ind_for_conformer: tmol.types.torch.Tensor[torch.int64][:], n_dof_atoms_offset_for_conformer: tmol.types.torch.Tensor[torch.int64][:], conformer_built_by_sampler: tmol.types.torch.Tensor[torch.bool][:], conf_inds_for_sampler: tmol.types.torch.Tensor[torch.int64][:], sampler_n_rots_for_gbt: tmol.types.torch.Tensor[torch.int32][:], sampler_gbt_for_rotamer: tmol.types.torch.Tensor[torch.int32][:], sample_dict: dict, conf_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9])


