tmol.kinematics.metadata
========================

.. py:module:: tmol.kinematics.metadata


Classes
-------

.. autoapisummary::

   tmol.kinematics.metadata.DOFTypes
   tmol.kinematics.metadata.DOFMetadata


Module Contents
---------------

.. py:class:: DOFTypes

   Bases: :py:obj:`enum.IntEnum`


   .. rubric:: Docstring

   .. code-block:: text

      High-level class of kinematic DOF types.
      

   .. py:attribute:: jump
      :value: 0



   .. py:attribute:: bond_angle


   .. py:attribute:: bond_distance


   .. py:attribute:: bond_torsion


.. py:class:: DOFMetadata

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      The location, type, and descriptive ids of valid dofs within a KinForest.
      
      Descriptive entries for dofs within a KinForest, this provides a 1-d
      structure to select and report a subset of entries within a KinDOF buffer.
      DOFMetadata sets are used to indicate mobile vs fixed dofs for KinematicOp
      dof to coordinate functions.
      
      DOFMetadata supports isomorphic conversion between a DataFrame and
      TensorGroup representation to support symbolic selection. This converts the
      IntEnum encoded "dof_type" entry into a string categorical column.
      
      The DOFMetadata data members, just like the KinForest, suffer from the same
      confusion about what an index represents because there are two ways to index
      the data:
      
      - The "Target Order" (TO) that refers to the index of an atom in the PoseStack
        it came from where the coordinate tensor is squashed to (N,3)
      - The "KinForest Order" (KFO) that refers to the order that an atom's node appears
        in the KinForest; this second ordering puts the index of any child atom after
        the index for any parent atom
      
      The DOFMetadata class indexes all available DOFs in the system. There are 9 possible
      DOFs per atom (either 3 for BondedAtoms or 9 for JumpAtoms), but in actuality,
      there are many fewer valid DOFs. The DOFMetadata class indexes valid DOFs.
      
      For each valid DOF i, there's:
      - node_idx[i]: the KFO index of the atom that DOF i belongs to
      - dof_idx[i]: the index between 0-8 for DOF i on its atom
      - dof_type[i]: the DOF type (either a BondDOFType or a JumpDOFType) for DOF i
      - parent_id[i]: the TO index for the parent to node_idx[i] for DOF i
      - child_id[i]: the TO index for node_idx[i] for DOF i
      
      The DOFMetadata class is primarily used to index into torch tensors in python,
      and therefore all of its dtypes are 64-bit integers.
      
      

   .. py:attribute:: node_idx
      :type:  tmol.types.torch.Tensor[torch.long][Ellipsis]


   .. py:attribute:: dof_idx
      :type:  tmol.types.torch.Tensor[torch.long][Ellipsis]


   .. py:attribute:: dof_type
      :type:  tmol.types.torch.Tensor[torch.long][Ellipsis]


   .. py:attribute:: parent_id
      :type:  tmol.types.torch.Tensor[torch.long][Ellipsis]


   .. py:attribute:: child_id
      :type:  tmol.types.torch.Tensor[torch.long][Ellipsis]


   .. py:method:: for_kinforest(kinforest: tmol.kinematics.datatypes.KinForest)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Return all valid dofs within a KinForest.
         


   .. py:method:: to_frame() -> pandas.DataFrame


   .. py:method:: from_frame(frame)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Convert from DataFrame to metadata, discarding any unneeded columns.
         


