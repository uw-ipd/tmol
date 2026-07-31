tmol.pack.packer_task
=====================

.. py:module:: tmol.pack.packer_task


Classes
-------

.. autoapisummary::

   tmol.pack.packer_task.PackerPalleteAnnotation
   tmol.pack.packer_task.PackerPalette
   tmol.pack.packer_task.PackerTask
   tmol.pack.packer_task.SetPackerTask


Functions
---------

.. autoapisummary::

   tmol.pack.packer_task.set_compare


Module Contents
---------------

.. py:function:: set_compare(x, y)

   .. rubric:: Docstring

   .. code-block:: text

      Treat the collections x and y as if they are sets. Return true if they
      contain the same elements and false otherwise
      

.. py:class:: PackerPalleteAnnotation

   .. py:attribute:: max_n_allowed
      :type:  int


   .. py:attribute:: n_allowed_block_types_for_block_type
      :type:  tmol.types.torch.Tensor[torch.int64][:]


   .. py:attribute:: allowed_block_types_for_block_type
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: allowed_block_type_is_orig
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: restrict_to_repacking_masks
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


.. py:class:: PackerPalette

   .. py:method:: block_types_from_original(pbt: tmol.pose.packed_block_types.PackedBlockTypes, orig: tmol.types.torch.Tensor[torch.int64][:, :]) -> tuple[tmol.types.torch.Tensor[torch.int64][:, :], tmol.types.torch.Tensor[torch.int64][:, :, :], tmol.types.torch.Tensor[torch.bool][:, :, :]]


   .. py:method:: create_restrict_to_repacking_mask(pbt: tmol.pose.packed_block_types.PackedBlockTypes, orig: tmol.types.torch.Tensor[torch.int64][:, :])


   .. py:method:: default_conformer_samplers()

      .. rubric:: Docstring

      .. code-block:: text

         All positions must build one rotamer, even if they are not being optimized.
         
         Each block must have coordinates represented in the tensor with the other
         rotamers, and the easiest way to do that is to create a rotamer with the
         DOFs of the input conformation. The FallbackSampler copies these DOFs
         from the inverse-folded coordinates of the starting Pose's blocks, but
         only for positions where no other sampler provides rotamers (e.g. residue
         types not covered by DunbrackChiSampler). Positions with at least one
         other sampler are left to that sampler exclusively.
         Future versions of PackerPalette have the option to override this method.
         


.. py:class:: PackerTask(systems: tmol.pose.pose_stack.PoseStack, palette: PackerPalette)

   .. py:attribute:: pbt


   .. py:attribute:: device


   .. py:attribute:: is_real_block


   .. py:attribute:: per_block_orig_block_type


   .. py:attribute:: restrict_to_repacking_masks


   .. py:attribute:: per_block_is_block_type_allowed


   .. py:attribute:: conformer_samplers


   .. py:attribute:: conformer_sampler_index


   .. py:attribute:: per_block_conformer_sampler_allowed


   .. py:attribute:: per_block_chi_expansion


   .. py:method:: restrict_to_repacking()


   .. py:method:: restrict_absent_name3s(name3s)

      .. rubric:: Docstring

      .. code-block:: text

         Disallow all block types at all positions except those with the given name3s.
         
         This is somewhat slow and does not cache the relationship between name3s and
         permitted block types, so consider writing your own version of this function
         if you call with the same list of name3s many times.
         


   .. py:method:: add_conformer_sampler(sampler: tmol.pack.rotamer.conformer_sampler.ConformerSampler)


   .. py:method:: add_conformer_sampler_by_block_mask(sampler: tmol.pack.rotamer.conformer_sampler.ConformerSampler, block_type_mask: tmol.types.torch.Tensor[torch.bool][:, :])


   .. py:method:: or_expand_chi(chi_ind: int)


   .. py:method:: or_expand_chi_to(chi_ind: int, sample_level: int)


   .. py:method:: disable_packing_by_block_mask(block_type_mask: tmol.types.torch.Tensor[torch.bool][:, :])


   .. py:method:: or_bump_check(setting=True)


   .. py:property:: bump_check

      .. rubric:: Docstring

      .. code-block:: text

         bump_check eliminates rotamers from consideration if
         they have a high interaction energy with "the background,"
         which is computed by taking the best energy a rotamer has
         with each neighbor across all the neighbor's rotamers.
         
         bump_check removes ~40% of all rotamers and can significantly
         improve running time, but, this comes at the expense of
         eliminating rotamers that are sometimes the best option
         when all others are bad in ways that bump-check's rosie
         estimation cannot predict. In ~10% of tested crystal
         structures packed with only the base rotamers (no ex flags),
         bump_check increased the energy of the final rotamer
         rotamer assignment by >20 kcal/mol.
         
         bump_check's logic: eliminate a rotamer if it has a
         best possible energy with its neighbors and itself
         >5 kcal/mol and at least one other rotamer of the same
         block type at that residue has an energy less than
         5 kcal/mol.


.. py:class:: SetPackerTask

   .. rubric:: Docstring

   .. code-block:: text

      Set as in concrete. Once everything wrt the desired packing
      task has been determined, pack_rotamers will construct this
      object to create and hold the many mappings that the various
      members of the packer need.
      

   .. py:method:: from_packer_task(task: PackerTask)
      :classmethod:



