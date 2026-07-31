tmol.kinematics.move_map
========================

.. py:module:: tmol.kinematics.move_map


Classes
-------

.. autoapisummary::

   tmol.kinematics.move_map.CartesianMoveMap
   tmol.kinematics.move_map.MoveMap
   tmol.kinematics.move_map.MinimizerMap


Module Contents
---------------

.. py:class:: CartesianMoveMap

   .. rubric:: Docstring

   .. code-block:: text

      Move map for Cartesian-space minimization.
      
      A lightweight wrapper around a coordinate mask that can be passed as the
      ``move_map`` argument to :func:`~tmol.relax.fast_relax.fast_relax` when
      using a Cartesian ``min_fn`` (e.g. one built around
      :func:`~tmol.optimization.minimizers.run_cart_min`).
      
      Unlike the full :class:`MoveMap`, which describes freedom in internal
      (torsion/jump) DOF space, ``CartesianMoveMap`` works directly in atomic
      Cartesian space.  The ``min_fn`` is responsible for extracting
      ``coord_mask`` and passing it to
      :class:`~tmol.optimization.sfxn_modules.CartesianSfxnNetwork`.
      
      :param coord_mask: Boolean tensor of shape ``[n_poses, max_n_atoms]``
                         indicating which atoms are free to move.  ``None`` means all
                         atoms are free to move (the default behaviour of
                         :class:`~tmol.optimization.sfxn_modules.CartesianSfxnNetwork`).
      

   .. py:attribute:: coord_mask
      :type:  Optional[torch.Tensor]
      :value: None



.. py:class:: MoveMap(n_poses: int, max_n_blocks: int, max_n_named_torsions: int, max_n_atoms_per_block: int, device: Optional[torch.device] = None)

   .. py:attribute:: move_all_jumps
      :type:  bool


   .. py:attribute:: move_all_root_jumps
      :type:  bool


   .. py:attribute:: move_all_mc
      :type:  bool


   .. py:attribute:: move_all_sc
      :type:  bool


   .. py:attribute:: move_all_named_torsions
      :type:  bool


   .. py:attribute:: non_ideal
      :type:  bool


   .. py:attribute:: move_jumps
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: move_jumps_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: move_root_jumps
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: move_root_jumps_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: move_mcs
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: move_mcs_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: move_scs
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: move_scs_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: move_named_torsions
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: move_named_torsions_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: move_mc
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :]


   .. py:attribute:: move_mc_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :]


   .. py:attribute:: move_sc
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :]


   .. py:attribute:: move_sc_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :]


   .. py:attribute:: move_named_torsion
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :]


   .. py:attribute:: move_named_torsion_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :]


   .. py:attribute:: move_jump_dof
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :]


   .. py:attribute:: move_jump_dof_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :]


   .. py:attribute:: move_root_jump_dof
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :]


   .. py:attribute:: move_root_jump_dof_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :]


   .. py:attribute:: move_atom_dof
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :, :]


   .. py:attribute:: move_atom_dof_mask
      :type:  tmol.types.torch.Tensor[torch.bool][:, :, :, :]


   .. py:method:: from_pose_stack(ps: tmol.pose.pose_stack.PoseStack)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Main construction utility for MoveMap.
         


   .. py:method:: set_move_all_jump_dofs_for_jump(pose_selection: Union[tmol.types.torch.Tensor, int], jump_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)

      .. rubric:: Docstring

      .. code-block:: text

         Enable or disable all jump dofs for a particular set of jumps on particular poses.
         
         If jump_selection is None, then the two dimensional settings tensor will be indexed by
         the pose_selection tensor only; if both are not None, then the settings tensor will be indexed by
         both the pose_selection tensor and the jump_selection tensor.
         


   .. py:method:: set_move_all_jump_dofs_for_root_jump(pose_selection: Union[tmol.types.torch.Tensor, int], root_jump_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)

      .. rubric:: Docstring

      .. code-block:: text

         Enable or disable all jump dofs for a particular set of root-jumps on particular poses.
         
         If root_jump_selection is None, then the two dimensional settings tensor will be indexed by
         the pose_selection tensor only; if both are not None, then the settings tensor will be indexed by
         both the pose_selection tensor and the jump_selection tensor.
         


   .. py:method:: set_move_all_mc_tors_for_blocks(pose_selection: Union[tmol.types.torch.Tensor, int], block_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)

      .. rubric:: Docstring

      .. code-block:: text

         Enable or disable all DOFs for a partiular set of blocks on particular poses.
         
         If block_selection is None, then the two dimensional settings tensor will be indexed by
         the pose_selection tensor only; if both are not None, then the tensor will be indexed by
         the pose_selection tensor and the block_selection tensor.
         
         Valid combinations of pose_selection and block_selection are, e.g.:
           - pose_selection: int, block_selection: int == a particular block on a particular pose
           - pose_selection: int, block_selection: None == all blocks on a particular pose
           - pose_selection: Tensor[bool][n_poses, max_n_blocks], block_selection: None == pose/block pairs encoded in "pose_selection" tensor
           - pose_selection: Tensor[int][N], block_selection: Tensor[int][N] == different blocks on different poses, selected by index
         


   .. py:method:: set_move_all_sc_tors_for_blocks(pose_selection: Union[tmol.types.torch.Tensor, int], block_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)


   .. py:method:: set_move_all_named_torsions_for_blocks(pose_selection: Union[tmol.types.torch.Tensor, int], block_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)


   .. py:method:: set_move_mc_tor_for_blocks(pose_selection: Union[tmol.types.torch.Tensor, int], block_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, tor_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)

      .. rubric:: Docstring

      .. code-block:: text

         Enable or disable partiular main-chain torsions for a particular set of blocks on particular poses.
         
         Valid combinations of block_selection and tor_selection are:
           - pose_selection: int, block_selection: int, tor_selection: int == a single DOF
           - pose_selection: Tensor[bool][n_poses, max_n_blocks, max_n_dofs], block_selection: None, tor_selection: None == pose/block/tor triples encoded in "pose_selection" tensor
           - pose_selection: Tensor[int][N], block_selection: Tensor[int][N], dof_selection: Tensor[int][N] == different torsions on different blocks on different poses, selected by index
         


   .. py:method:: set_move_sc_tor_for_blocks(pose_selection: Union[tmol.types.torch.Tensor, int], block_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, dof_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)

      .. rubric:: Docstring

      .. code-block:: text

         Enable or disable partiular side-chain torsions for a particular set of blocks on particular poses.
         


   .. py:method:: set_move_named_torsion_for_blocks(pose_selection: Union[tmol.types.torch.Tensor, int], block_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, tor_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)

      .. rubric:: Docstring

      .. code-block:: text

         Enable or disable partiular named-torsions for a particular set of blocks on particular poses.
         


   .. py:method:: set_move_jump_dof_for_jumps(pose_selection: Union[tmol.types.torch.Tensor, int], jump_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, dof_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)

      .. rubric:: Docstring

      .. code-block:: text

         Enable or disable all jump dofs for a particular set of jumps on particular poses.
         
         If jump_selection is None, then the two dimensional settings tensor will be indexed by
         the pose_selection tensor only; if both are not None, then the settings tensor will be indexed by
         both the pose_selection tensor and the jump_selection tensor.
         


   .. py:method:: set_move_jump_dof_for_root_jumps(pose_selection: Union[tmol.types.torch.Tensor, int], root_jump_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, dof_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)

      .. rubric:: Docstring

      .. code-block:: text

         Enable or disable all jump dofs for a particular set of root-jumps on particular poses.
         
         If root_jump_selection is None, then the two dimensional settings tensor will be indexed by
         the pose_selection tensor only; if both are not None, then the settings tensor will be indexed by
         both the pose_selection tensor and the root_jump_selection tensor.
         


   .. py:method:: set_move_atom_dof_for_blocks(pose_selection: Union[tmol.types.torch.Tensor, int], block_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, atom_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, dof_selection: Optional[Union[tmol.types.torch.Tensor, int]] = None, value: bool = True)

      .. rubric:: Docstring

      .. code-block:: text

         Enable or disable partiular atom dofs for a particular set of blocks on particular poses.
         
         Either only "pose_selection" should be not None or all four "selection" variables should be
         not None; in the former case, the settings tensor, self.move_atom_dof, will be indexed
         solely by the pose_selection tensor; in the latter case, the settings tensor will be indexed
         by all four selection tensors.
         
         This function offers the finest grain control over which dofs should be minimized and
         settings made using this function will override any settings made using the other
         settings tensors.
         


   .. py:property:: n_poses


   .. py:property:: max_n_jumps


   .. py:property:: max_n_blocks


   .. py:property:: max_n_named_torsions


   .. py:property:: max_n_atoms_per_block


.. py:class:: MinimizerMap(pose_stack: tmol.pose.pose_stack.PoseStack, kmd: tmol.kinematics.datatypes.KinematicModuleData, mm: MoveMap)

   .. py:attribute:: dof_mask


