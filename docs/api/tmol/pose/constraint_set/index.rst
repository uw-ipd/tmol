tmol.pose.constraint_set
========================

.. py:module:: tmol.pose.constraint_set


Classes
-------

.. autoapisummary::

   tmol.pose.constraint_set.ConstraintSet


Module Contents
---------------

.. py:class:: ConstraintSet

   .. py:attribute:: MAX_N_ATOMS
      :value: 4



   .. py:attribute:: device
      :type:  torch.device


   .. py:attribute:: n_poses
      :type:  int


   .. py:attribute:: constraint_function_inds
      :type:  tmol.types.torch.Tensor[torch.int][:]


   .. py:attribute:: constraint_atoms
      :type:  tmol.types.torch.Tensor[torch.int][:, 4, 3]


   .. py:attribute:: constraint_params
      :type:  tmol.types.torch.Tensor[torch.float32][:, :]


   .. py:attribute:: constraint_num_unique_blocks
      :type:  tmol.types.torch.Tensor[torch.int][:]


   .. py:attribute:: constraint_unique_blocks
      :type:  tmol.types.torch.Tensor[torch.int][:, :]


   .. py:attribute:: constraint_functions
      :type:  Tuple


   .. py:method:: create_empty(device: torch.device, n_poses: int) -> ConstraintSet
      :classmethod:



   .. py:method:: concatenate(constraint_sets: Tuple[Optional[ConstraintSet], Ellipsis], from_multiple_pose_stacks: bool = True, n_poses: Optional[int] = None, ps_offset: Optional[tmol.types.torch.Tensor[torch.int64][:]] = None) -> Optional[ConstraintSet]
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Concatenate multiple ConstraintSets into a single ConstraintSet.
         
         This function is particularly useful if you're creating a PoseStack from multiple
         PoseStacks, each of which has its own ConstraintSet. In that case, n_poses
         and ps_offset will be readily available. In this use case, "from_multiple_pose_stacks"
         should be set to True.
         
         The other use case is in creating multiple types of constraints for a single
         PoseStack and then combining them in a single go. This will be more efficient
         than repeatedly invoking add_constraints() as it skips the N^2 copy operations.
         In this use case, "from_multiple_pose_stacks" should be set to False.
         


   .. py:method:: clone() -> ConstraintSet


   .. py:method:: to(device: torch.device) -> ConstraintSet


   .. py:method:: split(index) -> ConstraintSet

      .. rubric:: Docstring

      .. code-block:: text

         Split out a single pose's worth of constraints from a batch.
         


   .. py:method:: count_unique_blocks(atom_indices)


   .. py:method:: add_constraints_to_all_poses(fn, atom_indices, params=None) -> ConstraintSet

      .. rubric:: Docstring

      .. code-block:: text

         If all Poses in the PoseStack should be constrained in the same way, then
         this convenience function will take a list of atom indices for a single Pose
         and replicate them across all the Poses in the PoseStack.
         


   .. py:method:: add_constraints(fn, atom_indices, params=None) -> ConstraintSet

      .. rubric:: Docstring

      .. code-block:: text

         Create a new ConstraintSet that includes all the old constraints plus the new ones.
         
         atom_indices: either (n_constraints, n_atoms, 3) or (n_constraints, n_atoms, 2)
                       If the latter, the constraint will be applied to all poses
         


   .. py:method:: replicate_constraints(n_poses, c_atms, c_params)
      :classmethod:



