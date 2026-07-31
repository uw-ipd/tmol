tmol.pack.rotamer.single_residue_kinforest
==========================================

.. py:module:: tmol.pack.rotamer.single_residue_kinforest


Classes
-------

.. autoapisummary::

   tmol.pack.rotamer.single_residue_kinforest.RotamerKintree
   tmol.pack.rotamer.single_residue_kinforest.PackedRotamerKintree


Functions
---------

.. autoapisummary::

   tmol.pack.rotamer.single_residue_kinforest.construct_single_residue_kinforest
   tmol.pack.rotamer.single_residue_kinforest.coalesce_single_residue_kinforests


Module Contents
---------------

.. py:class:: RotamerKintree

   .. py:attribute:: kinforest_idx
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: id
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: doftype
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: parent
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: frame_x
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: frame_y
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: frame_z
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: nodes
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: scans
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: gens
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: n_scans_per_gen
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: dofs_ideal
      :type:  tmol.types.array.NDArray[numpy.int32][:]


.. py:class:: PackedRotamerKintree

   .. py:attribute:: kinforest_idx
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: id
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: doftype
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: parent
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: frame_x
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: frame_y
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: frame_z
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: n_nodes
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: nodes
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: scans
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: gens
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: n_scans_per_gen
      :type:  tmol.types.array.NDArray[numpy.int32][:, :]


   .. py:attribute:: dofs_ideal
      :type:  tmol.types.torch.Tensor[torch.float32][:, :]


.. py:function:: construct_single_residue_kinforest(restype: tmol.chemical.restypes.RefinedResidueType)

.. py:function:: coalesce_single_residue_kinforests(pbt: tmol.pose.packed_block_types.PackedBlockTypes)

