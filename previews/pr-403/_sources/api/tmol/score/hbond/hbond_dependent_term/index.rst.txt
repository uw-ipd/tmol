tmol.score.hbond.hbond_dependent_term
=====================================

.. py:module:: tmol.score.hbond.hbond_dependent_term


Classes
-------

.. autoapisummary::

   tmol.score.hbond.hbond_dependent_term.HBondBlockTypeParams
   tmol.score.hbond.hbond_dependent_term.HBondPackedBlockTypesParams
   tmol.score.hbond.hbond_dependent_term.HBondDependentTerm


Functions
---------

.. autoapisummary::

   tmol.score.hbond.hbond_dependent_term.attached_H_for_don


Module Contents
---------------

.. py:function:: attached_H_for_don(atom_is_hydrogen, D_idx, bonds, bond_spans)

.. py:class:: HBondBlockTypeParams

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: donH_inds
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: don_hvy_inds
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: acc_inds
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: tile_n_donH
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: tile_n_don_hvy
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: tile_n_acc
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: tile_donH_inds
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: tile_donH_hvy_inds
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: tile_don_hvy_inds
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: tile_which_donH_of_donH_hvy
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: tile_acc_inds
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: tile_donorH_type
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: tile_acceptor_type
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: tile_acceptor_hybridization
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: tile_acceptor_n_attached_H
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: is_hydrogen
      :type:  tmol.types.array.NDArray[numpy.int32][:]


.. py:class:: HBondPackedBlockTypesParams

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: tile_n_donH
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: tile_n_don_hvy
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: tile_n_acc
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: tile_donH_inds
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: tile_donH_hvy_inds
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: tile_don_hvy_inds
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: tile_which_donH_of_donH_hvy
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: tile_acc_inds
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: tile_donorH_type
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: tile_acceptor_type
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: tile_acceptor_hybridization
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: tile_acceptor_n_attached_H
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: is_hydrogen
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


.. py:class:: HBondDependentTerm(param_db: tmol.database.ParameterDatabase, device: torch.device)

   Bases: :py:obj:`tmol.score.bond_dependent_term.BondDependentTerm`


   .. py:attribute:: atom_type_resolver
      :type:  tmol.score.chemical_database.AtomTypeParamResolver


   .. py:attribute:: hbond_database
      :type:  tmol.database.scoring.hbond.HBondDatabase


   .. py:attribute:: hbond_resolver
      :type:  tmol.score.hbond.params.HBondParamResolver


   .. py:attribute:: device
      :type:  torch.device


   .. py:attribute:: tile_size
      :value: 32



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
         


