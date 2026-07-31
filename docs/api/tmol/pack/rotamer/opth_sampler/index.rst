tmol.pack.rotamer.opth_sampler
==============================

.. py:module:: tmol.pack.rotamer.opth_sampler


Classes
-------

.. autoapisummary::

   tmol.pack.rotamer.opth_sampler.OptHSamplerRTCache
   tmol.pack.rotamer.opth_sampler.OptHSamplerPackedBlockTypeCache
   tmol.pack.rotamer.opth_sampler.OptHSampler


Module Contents
---------------

.. py:class:: OptHSamplerRTCache

   .. rubric:: Docstring

   .. code-block:: text

      Per-residue-type annotation for OptHSampler.
      
      Covers two orthogonal features:
      1. Proton chi sampling (SER/THR/TYR/CYS): samples the terminal (proton)
         chi angle using values from restype definition.
      2. NHQ flip (ASN/GLN/HIS/HIS_D): generates the input conformation plus a
         180-degree rotation about the last chi angle.
         HIS additionally generates both protonation states.
      

   .. py:attribute:: has_proton_chi
      :type:  bool


   .. py:attribute:: n_chi_total
      :type:  int


   .. py:attribute:: chi_defining_atom
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: n_proton_samples
      :type:  int


   .. py:attribute:: expanded_samples
      :type:  tmol.types.array.NDArray[numpy.float32][:, :]


   .. py:attribute:: n_samples_per_chi
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: nhq_chi_col
      :type:  int


   .. py:attribute:: nhq_chi_atom
      :type:  int


   .. py:attribute:: nhq_chi_4atoms
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: nhq_downstream_kfo
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: is_his
      :type:  bool


.. py:class:: OptHSamplerPackedBlockTypeCache

   .. py:attribute:: opth_sample_for_bt
      :type:  tmol.types.torch.Tensor[torch.bool][:]


   .. py:attribute:: has_proton_chi
      :type:  tmol.types.torch.Tensor[torch.bool][:]


   .. py:attribute:: n_chi_total
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: chi_defining_atom
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: n_proton_samples
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: expanded_samples
      :type:  tmol.types.torch.Tensor[torch.float32][:, :, :]


   .. py:attribute:: n_samples_per_chi
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: nhq_chi_col
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: nhq_chi_atom
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: nhq_chi_4atoms
      :type:  tmol.types.torch.Tensor[torch.int32][:, 4]


   .. py:attribute:: nhq_downstream_kfo
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: is_his
      :type:  tmol.types.torch.Tensor[torch.bool][:]


   .. py:attribute:: n_samples_for_bt_by_orig_bt
      :type:  tmol.types.torch.Tensor[torch.int32][2, :, :]


   .. py:attribute:: n_chi_needed_for_bt
      :type:  tmol.types.torch.Tensor[torch.int32][2, :]


.. py:class:: OptHSampler

   Bases: :py:obj:`tmol.pack.rotamer.conformer_sampler.ConformerSampler`


   .. rubric:: Docstring

   .. code-block:: text

      Build rotamers by sampling proton chi angles only, keeping all heavy
      atoms at their input-conformation positions.
      
      When flip_NHQ is True (default), also builds flip rotamers for:
      - ASN/GLN: current conformation + 180-degree rotation of the last chi.
      - HIS/HIS_D: {HIS, HIS_D} x {current chi2, chi2+180} = 4 rotamers.
        All atoms through CG are taken from the input; ring atoms are rebuilt
        from ideal geometry for three non-input variants.
      
      NOTE: DunbrackChiSampler and OptHSampler must not be assigned to the
      same block (Dunbrack already samples proton chis, so both on one block
      oversamples). Assigning them to different blocks in the same task is fine.
      

   .. py:attribute:: flip_NHQ
      :type:  bool
      :value: True



   .. py:method:: sampler_name()
      :classmethod:



   .. py:method:: defines_rotamers_for_rt(rt: tmol.chemical.restypes.RefinedResidueType)


   .. py:method:: defines_rotamers_for_bts(pbt: tmol.pose.packed_block_types.PackedBlockTypes, bt_inds: tmol.types.torch.Tensor[torch.int64]) -> tmol.types.torch.Tensor[torch.bool]


   .. py:method:: first_sc_atoms_for_rt(rt: tmol.chemical.restypes.RefinedResidueType) -> Tuple[str, Ellipsis]


   .. py:method:: create_samples_for_poses(pose_stack: tmol.pose.pose_stack.PoseStack, task: SetPackerTask) -> Tuple[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int32][:], dict]


   .. py:method:: fill_dofs_for_samples(pose_stack: tmol.pose.pose_stack.PoseStack, task: PackerTask, orig_kinforest: tmol.kinematics.datatypes.KinForest, orig_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9], gbt_for_conformer: tmol.types.torch.Tensor[torch.int64][:], block_type_ind_for_conformer: tmol.types.torch.Tensor[torch.int64][:], n_dof_atoms_offset_for_conformer: tmol.types.torch.Tensor[torch.int64][:], conformer_built_by_sampler: tmol.types.torch.Tensor[torch.bool][:], conf_inds_for_sampler: tmol.types.torch.Tensor[torch.int64][:], sampler_n_rots_for_gbt: tmol.types.torch.Tensor[torch.int32][:], sampler_gbt_for_rotamer: tmol.types.torch.Tensor[torch.int32][:], sample_dict: dict, conf_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9])


