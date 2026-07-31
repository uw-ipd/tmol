tmol.score.cartbonded.cartbonded_energy_term
============================================

.. py:module:: tmol.score.cartbonded.cartbonded_energy_term


Attributes
----------

.. autoapisummary::

   tmol.score.cartbonded.cartbonded_energy_term.debug


Classes
-------

.. autoapisummary::

   tmol.score.cartbonded.cartbonded_energy_term.CartBondedBlockAnnotations
   tmol.score.cartbonded.cartbonded_energy_term.CartBondedPackedBlockTypesAnnotations
   tmol.score.cartbonded.cartbonded_energy_term.CartBondedEnergyTerm


Module Contents
---------------

.. py:data:: debug
   :value: False


.. py:class:: CartBondedBlockAnnotations

   .. py:attribute:: cartbonded_subgraphs
      :type:  torch.Tensor


   .. py:attribute:: cartbonded_subgraph_type_counts
      :type:  torch.Tensor


   .. py:attribute:: cartbonded_subgraph_type_offsets
      :type:  torch.Tensor


   .. py:attribute:: cartbonded_params
      :type:  dict


.. py:class:: CartBondedPackedBlockTypesAnnotations

   .. py:attribute:: cartbonded_subgraphs
      :type:  torch.Tensor


   .. py:attribute:: cartbonded_subgraph_offsets
      :type:  torch.Tensor


   .. py:attribute:: cartbonded_subgraph_type_counts
      :type:  torch.Tensor


   .. py:attribute:: cartbonded_subgraph_type_offsets
      :type:  torch.Tensor


   .. py:attribute:: cartbonded_max_subgraphs_per_block
      :type:  int


   .. py:attribute:: cartbonded_atom_unique_id_index
      :type:  dict


   .. py:attribute:: cartbonded_params_hash_keys
      :type:  torch.Tensor


   .. py:attribute:: cartbonded_params_hash_values
      :type:  torch.Tensor


.. py:class:: CartBondedEnergyTerm(param_db: tmol.database.ParameterDatabase, device: torch.device)

   Bases: :py:obj:`tmol.score.atom_type_dependent_term.AtomTypeDependentTerm`


   .. py:attribute:: device
      :type:  torch.device


   .. py:attribute:: improper_roots
      :type:  set()


   .. py:attribute:: cart_database


   .. py:attribute:: hash


   .. py:method:: class_name()
      :classmethod:



   .. py:method:: score_types()
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Return the list of score types that this EnergyTerm computes
         
         The order that the term reports score types in this function should be
         the same order that it reports the scores themselves in the output
         tensor
         


   .. py:method:: n_bodies()

      .. rubric:: Docstring

      .. code-block:: text

         Return the number of residues that this term operates on
         
         1, 2, or -1 to represent the whole structure
         


   .. py:method:: find_subgraphs(bonds, block_type)


   .. py:method:: get_raw_params_for_res(res: str)


   .. py:method:: get_formatted_atoms_and_params(raw_params)


   .. py:method:: get_params_for_res(res: str)


   .. py:method:: setup_block_type(block_type: tmol.chemical.restypes.RefinedResidueType)

      .. rubric:: Docstring

      .. code-block:: text

         Make a one-time annotation on the block type. These annotations will
         probably require string comparison and may be slow; they should be
         performed only once, so the EnergyTerm must check that its annotation
         is not already present in the block type. Annotations should be in
         numpy data structures (and stored on the CPU).
         
         If the annotation requires more than one array, then the EnergyTerm
         should use a python class to store those arrays. E.g.,
         class FooSet:
             foo_array1: NDArray[numpy.int32][:]
             foo_array2: NDArray[numpy.int32][:, :]
         
         If the kind of annotation made depends on data that may change
         between different instances of the same term, then the annotation
         should be a map whose key is a function of the perhaps-changing
         data. The term should calculate that key at its construction to
         make retrieval efficient. (Any such data that sways how the
         calculation is made should never change over the lifetime of the
         instance; if new values for that data are needed a separate
         instance should be created.)
         


   .. py:method:: setup_packed_block_types(packed_block_types: tmol.pose.packed_block_types.PackedBlockTypes)

      .. rubric:: Docstring

      .. code-block:: text

         Make a one-time annotation of the packed-block types. This annotation
         should mostly involve concatenating the previously-made numpy annotations
         on the block types that the packed-block types contains. E.g. if the
         EnergyTerm annotates the block types with an i-dimensional array "foo,"
         then it should also annotate the PackedBlockTypes with an (i+1)-dimensional
         tensor "foo" where the first dimension will index across the different
         block types in foo in the order that those block types appear in the
         PackedBlockTypes' list of active block types. Sometimes the size of the
         i-dimensional arrays will differ between block types; the (i+1)-dimensional
         tensor should be dimensioned to the maximal size for each of the i dimensions
         among the set of dimensions of the various block types. The extra padding
         in such cases is recommended to be filled with a sentinel value of -1.
         
         As with the block type annotation, if more than one tensor is required,
         then the annotation should be a class. If the annotation is based on
         data that might differ between instances, then the annotation should be
         a map whose keys are determined by the data.
         
         The EnergyMethod should begin by checking that it has not already made
         this annotation. Any array data in the annotation should be torch
         tensors and should live on the PackedBlockTypes' device.
         


   .. py:method:: setup_poses(poses: tmol.pose.pose_stack.PoseStack)

      .. rubric:: Docstring

      .. code-block:: text

         Make a one-time annotation of a PoseStack. These annotations should
         not depend on anything about the conformation or block-type identity of
         the PoseStack, but can depend on the chemical connectivity, the number
         of poses in the stack, and the maximum number of atoms in the stack.
         
         Any array data should be stored in torch tensors and live on the
         pose_stack's device.
         


   .. py:method:: get_pose_score_term_function()


   .. py:method:: get_rotamer_score_term_function()


   .. py:method:: get_score_term_attributes(pose_stack)


