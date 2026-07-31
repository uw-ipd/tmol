tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler
===============================================

.. py:module:: tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler


Classes
-------

.. autoapisummary::

   tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler.DunSamplerRTCache
   tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler.DunSamplerPBTCache
   tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler.DunbrackChiSampler


Functions
---------

.. autoapisummary::

   tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler.create_dunbrack_sampler_from_database


Module Contents
---------------

.. py:class:: DunSamplerRTCache

   .. rubric:: Docstring

   .. code-block:: text

      Data to store in RefinedResidueType that will be reused
      repeatedly in the creation of the DunSamplerPBTCache
      

   .. py:attribute:: bbdihe_uaids
      :type:  tmol.types.array.NDArray[numpy.int32][2, 4, 3]


   .. py:attribute:: chi_defining_atom
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: non_dunbrack_sample_counts
      :type:  tmol.types.array.NDArray[numpy.int32][:, 2]


   .. py:attribute:: non_dunbrack_samples
      :type:  tmol.types.array.NDArray[numpy.int32][:, 2, :]


   .. py:attribute:: rottable_set_for_bt
      :type:  int


.. py:class:: DunSamplerPBTCache

   .. rubric:: Docstring

   .. code-block:: text

      Data needed for chi sampling and for reporting how
      the chi are to be assigned to atoms
      

   .. py:attribute:: bbdihe_uaids
      :type:  tmol.types.torch.Tensor[torch.int32][:, 2, 4, 3]


   .. py:attribute:: chi_defining_atom
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: non_dunbrack_sample_counts
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, 2]


   .. py:attribute:: non_dunbrack_samples
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, 2, :]


   .. py:attribute:: defines_rotamers_for_bts
      :type:  tmol.types.torch.Tensor[torch.bool][:]


   .. py:attribute:: rottable_set_for_bt
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:property:: max_n_chi


.. py:class:: DunbrackChiSampler(dun_param_resolver: tmol.score.dunbrack.params.DunbrackParamResolver)

   Bases: :py:obj:`tmol.pack.rotamer.chi_sampler.ChiSampler`


   .. py:attribute:: dun_param_resolver
      :type:  tmol.score.dunbrack.params.DunbrackParamResolver


   .. py:property:: device


   .. py:method:: from_database(param_resolver: tmol.score.dunbrack.params.DunbrackParamResolver)
      :classmethod:



   .. py:method:: sampler_name()
      :classmethod:



   .. py:method:: annotate_residue_type(restype: tmol.chemical.restypes.RefinedResidueType)

      .. rubric:: Docstring

      .. code-block:: text

         TEMP TEMP TEMP: assume the dihedrals we care about are phi and psi
         


   .. py:method:: annotate_packed_block_types(packed_block_types: tmol.pose.packed_block_types.PackedBlockTypes)


   .. py:method:: defines_rotamers_for_rt(rt: tmol.chemical.restypes.RefinedResidueType)


   .. py:method:: defines_rotamers_for_bts(pbt: tmol.pose.packed_block_types.PackedBlockTypes, bt_inds: tmol.types.torch.Tensor[torch.int64]) -> tmol.types.torch.Tensor[torch.bool]


   .. py:method:: first_sc_atoms_for_rt(rt: tmol.chemical.restypes.RefinedResidueType) -> Tuple[str, Ellipsis]


   .. py:method:: sample_chi_for_poses(pose_stack: tmol.pose.pose_stack.PoseStack, task: tmol.pack.packer_task.SetPackerTask) -> Tuple[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.float32][:, :]]


   .. py:method:: atom_indices_for_backbone_dihedral(pose_stack: tmol.pose.pose_stack.PoseStack, bb_dihedral_ind: int)


   .. py:method:: launch_rotamer_building(coords, ndihe_for_res, dihedral_offset_for_res, dihedral_atom_inds, bubl_and_rottable_set_for_buildable_restype, chi_expansion_for_buildable_restype, non_dunbrack_expansion_for_buildable_restype, non_dunbrack_expansion_counts_for_buildable_restype, prob_cumsum_limit_for_buildable_restype, nchi_for_buildable_restype)


   .. py:method:: package_samples_for_output(pbt: tmol.pose.packed_block_types.PackedBlockTypes, task: tmol.pack.packer_task.SetPackerTask, n_gbt_total: int, bbt_to_gbt: tmol.types.torch.Tensor[torch.int64][:], block_type_ind_for_brt: tmol.types.torch.Tensor[torch.int64][:], max_n_chi: int, sampled_chi)


.. py:function:: create_dunbrack_sampler_from_database(param_db: tmol.database.ParameterDatabase, device: torch.device) -> DunbrackChiSampler

   .. rubric:: Docstring

   .. code-block:: text

      Create a DunbrackChiSampler from the default database.
      
      :param param_db: The parameter database containing Dunbrack parameters
      :param device: The device to use for the sampler
      
      :returns: Configured sampler for rotamer building
      :rtype: DunbrackChiSampler
      

