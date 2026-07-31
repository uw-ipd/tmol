tmol.pack.rotamer.build_rotamers
================================

.. py:module:: tmol.pack.rotamer.build_rotamers


Functions
---------

.. autoapisummary::

   tmol.pack.rotamer.build_rotamers.correct_phi_c_for_jump_parents
   tmol.pack.rotamer.build_rotamers.exc_cumsum_from_inc_cumsum
   tmol.pack.rotamer.build_rotamers.annotate_restype
   tmol.pack.rotamer.build_rotamers.annotate_packed_block_types
   tmol.pack.rotamer.build_rotamers.annotate_everything
   tmol.pack.rotamer.build_rotamers.update_nodes
   tmol.pack.rotamer.build_rotamers.update_scan_starts
   tmol.pack.rotamer.build_rotamers.construct_scans_for_conformers
   tmol.pack.rotamer.build_rotamers.load_from_rotamers
   tmol.pack.rotamer.build_rotamers.load_from_rotamers_w_offsets
   tmol.pack.rotamer.build_rotamers.load_rotamer_parents
   tmol.pack.rotamer.build_rotamers.construct_kinforest_for_conformers
   tmol.pack.rotamer.build_rotamers.measure_dofs_from_orig_coords
   tmol.pack.rotamer.build_rotamers.measure_pose_dofs
   tmol.pack.rotamer.build_rotamers.merge_conformer_samples
   tmol.pack.rotamer.build_rotamers.calculate_rotamer_coords
   tmol.pack.rotamer.build_rotamers.get_rotamer_origin_data
   tmol.pack.rotamer.build_rotamers.build_rotamers


Module Contents
---------------

.. py:function:: correct_phi_c_for_jump_parents(pbt, conformer_samples, new_ind_for_sampler_rotamer, block_type_ind_for_conformer_torch, n_atoms_offset_for_conformer_torch, conformer_kinforest, nodes, scans, gens, conf_dofs_kto)

   .. rubric:: Docstring

   .. code-block:: text

      For chi-defining atoms whose kinforest parent is a jump atom, the phi_c
      written by assign_chi_dofs_from_samples does not directly map to the
      chi dihedral angle measured from coordinates.  This function:
        1. Does a trial forward pass with the current DOFs.
        2. For each such atom, measures the actual dihedral from the trial coords.
        3. Adds (intended - measured) to conf_dofs_kto[atom_kto, 3] so the
           final forward pass produces the correct geometry.
      

.. py:function:: exc_cumsum_from_inc_cumsum(cumsum)

.. py:function:: annotate_restype(restype: tmol.chemical.restypes.RefinedResidueType, samplers: Tuple[tmol.pack.rotamer.chi_sampler.ChiSampler, Ellipsis], chem_db: tmol.database.chemical.ChemicalDatabase)

.. py:function:: annotate_packed_block_types(pbt: tmol.pose.packed_block_types.PackedBlockTypes)

.. py:function:: annotate_everything(chem_db: tmol.database.chemical.ChemicalDatabase, samplers: Tuple[tmol.pack.rotamer.chi_sampler.ChiSampler, Ellipsis], pbt: tmol.pose.packed_block_types.PackedBlockTypes)

.. py:function:: update_nodes(nodes_orig, genStartsStack, n_nodes_offset_for_rot, n_atoms_offset_for_rot)

   .. rubric:: Docstring

   .. code-block:: text

      Merge the 1-residue-kinforest nodes data so that all the rotamers can be
      built in a single generational-segmented-scan call. This has the structure
      of load-balanced search operation.
      

.. py:function:: update_scan_starts(n_scans, atomStartsOffsets, scanStartsStack, genStartsStack, ngenStack)

.. py:function:: construct_scans_for_conformers(pbt: tmol.pose.packed_block_types.PackedBlockTypes, block_type_ind_for_conf: tmol.types.array.NDArray[numpy.int32][:], n_atoms_for_conf: tmol.types.torch.Tensor[torch.int32][:], n_atoms_offset_for_conf: tmol.types.array.NDArray[numpy.int64][:])

.. py:function:: load_from_rotamers(arr: tmol.types.array.NDArray[numpy.int32][:, :], n_atoms_total: int, n_atoms_for_rot: tmol.types.array.NDArray[numpy.int32][:])

.. py:function:: load_from_rotamers_w_offsets(arr: tmol.types.array.NDArray[numpy.int32][:, :], n_atoms_total: int, n_atoms_for_rot: tmol.types.array.NDArray[numpy.int32][:], n_atoms_offset_for_rot: tmol.types.array.NDArray[numpy.int32][:])

.. py:function:: load_rotamer_parents(parents: tmol.types.array.NDArray[numpy.int32][:, :], n_atoms_total: int, n_atoms_for_rot: tmol.types.array.NDArray[numpy.int32][:], n_atoms_offset_for_rot: tmol.types.array.NDArray[numpy.int32][:])

.. py:function:: construct_kinforest_for_conformers(pbt: tmol.pose.packed_block_types.PackedBlockTypes, conf_block_type_ind: tmol.types.array.NDArray[numpy.int32][:], n_atoms_total: int, n_atoms_for_conf: tmol.types.torch.Tensor[torch.int32][:], block_offset_for_conf: tmol.types.array.NDArray[numpy.int64][:], device: torch.device)

   .. rubric:: Docstring

   .. code-block:: text

      Construct a KinForest for a set of conformers by stringing
      together the kinforest data for individual conformers.
      The "block_ofset_for_conf" array is used to construct
      the "id" tensor in the KinForest, which maps to the atom
      indices; thus it should contain the atom-index offsets
      for the first atom in each rotamer in the coords tensor
      that will be used to construct the kinforest_coords tensor.
      

.. py:function:: measure_dofs_from_orig_coords(coords: tmol.types.torch.Tensor[torch.float32][:, :, :], kinforest: tmol.kinematics.datatypes.KinForest)

.. py:function:: measure_pose_dofs(poses)

.. py:function:: merge_conformer_samples(conformer_samples) -> Tuple[tmol.types.torch.Tensor[torch.int64][:], tmol.types.torch.Tensor[torch.int64][:], tmol.types.torch.Tensor[torch.int64][:], List[tmol.types.torch.Tensor[torch.bool][:]], List[tmol.types.torch.Tensor[torch.int64][:]]]

   .. rubric:: Docstring

   .. code-block:: text

      Merge the lists of conformers as described by different conformer samplers.
      
      The conformer_samples variable is a list of tuples:
       - elem 0: Tensor[int][:] <-- the number of rotamers for each pose for each block for each block type
          where each buildable block type for each real residue is given a global index
       - elem 1: Tensor[int][:] <-- the global block-type index for each rotamer
       - elem 2+: Extra data that the chi sampler needs to preserve, where the first dimension
         is rotamer index based on elem 1's rotamer indices; the mapping from orig rotamer indices
         to merged rotamer indices will be constructed by this routine
      

.. py:function:: calculate_rotamer_coords(pbt: tmol.pose.packed_block_types.PackedBlockTypes, n_rots: int, n_atoms_total: int, rot_kinforest: tmol.kinematics.datatypes.KinForest, nodes: tmol.types.array.NDArray[numpy.int32][:], scans: tmol.types.array.NDArray[numpy.int32][:], gens: tmol.types.array.NDArray[numpy.int32][:], rot_dofs_kto: tmol.types.torch.Tensor[torch.float32][:, 9])

.. py:function:: get_rotamer_origin_data(task: tmol.pack.packer_task.SetPackerTask, gbt_for_rot: tmol.types.torch.Tensor[torch.int32][:])

.. py:function:: build_rotamers(poses: tmol.pose.pose_stack.PoseStack, task: tmol.pack.packer_task.SetPackerTask, chem_db: tmol.database.chemical.ChemicalDatabase)

