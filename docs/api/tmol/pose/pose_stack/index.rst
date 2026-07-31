tmol.pose.pose_stack
====================

.. py:module:: tmol.pose.pose_stack


Classes
-------

.. autoapisummary::

   tmol.pose.pose_stack.PoseStack


Module Contents
---------------

.. py:class:: PoseStack

   .. rubric:: Docstring

   .. code-block:: text

      The PoseStack class defines a batch (a stack) of molecular systems
      
      The PoseStack defines the per-residue chemistry, inter-residue
      connectivity, and coordinates of a set of molecular systems.
      The per-residue chemistry and the connectivity are meant to be
      constant over its lifetime; however, its coordinates are allowed
      to change. That is, a PoseStack may have its coords tensor written
      to without any negative consequences. Thus, you can minimize the
      coordinates in a PoseStack. The main way to change the chemistry or
      connectivity or even the ConstraintSet of a PoseStack is to use
      attr.evolve to return a completely new PoseStack but replacing
      the datamember(s) that you want to change. Such a PoseStack will be
      a shallow copy of the original PoseStack, so the coordinates tensor
      must always be cloned / replaced when evolving a PoseStack, or two
      PoseStacks may point to the same coordinates tensor, and then
      changes to the coordinates in one PoseStack would affect the other.
      
      Datamembers:
      packed_block_types: a representation of the chemical space that this
      PoseStack can contain. The PackedBlockTypes object aggregates a set
      of residue-types objects (RefinedResidueTypes) and holds annotations
      for this aggregate that must be made by the terms in the ScoreFunction
      in order for them to efficiently perform their calculations.
      
      coords: a tensor of [n_poses x max_n_atoms_per_pose x 3] holding the
      cartesian coordinates of the atoms in the system. The coordinates
      of the atoms are held in a contiguous array so that mixing very
      large residue types (e.g. heme) and very small residue types
      (e.g. water) does not waste memory / GPU cache.
      
      block_coord_offset: a tensor of [n_poses x max_n_residues] holding
      the starting indices in the coords tensor for the residues; offsets
      for custom kernels are 32-bit integers, offset for torch functions
      are 64-bit integers. We keep around both for performance reasons.
      
      inter_residue_connections: a tensor of
      [n_poses x max_n_residues x max_n_conn x 2] representing for each
      inter-residue connection point on each residue the 1) index of
      the residue it is connected to (sentinel of -1 for "no connection
      defined) and the connection-point index it is connected to
      (sentinel of -1, also).
      
      inter_block_bondsep: a integer tensor of shape
      [n_poses x max_n_residues x max_n_residues x max_n_conn x max_n_conn]
      stating the number of chemical bonds that separate every pair of
      inter-residue connections for every pair of residues -- up to a
      maximum inter-residue separation of
      tmol.chemical.MAX_SIG_BOND_SEPARATION (6 as of March 2024) --
      so that the number of chemical bonds separating arbitrary
      atom pairs may be rapidly computed for the interatomic energy
      calculations
      
      block_type_ind: the integer index for each block type (residue type)
      referring to the order in which that block type appears in the
      PoseStack's PackedBlockTypes object. A sentinel of -1 for positions
      where there is no block type.
      
      chain_id: the integer chain identifier for each residue
      
      pdb_info: a PDBInfo object holding the PDB-level information that's needed
      for writing out PDB / mmCIF files and keeping the original author labels
      for the chains and residues + occupancy and B-factor information for the atoms;
      none of these things are necessary for any structural manipulations or energy
      calculations, but they are invaluable for working with structures in any kind
      of pipeline.
      
      device: the torch.device that this collection of structures lives on
      

   .. py:attribute:: packed_block_types
      :type:  tmol.pose.packed_block_types.PackedBlockTypes


   .. py:attribute:: coords
      :type:  tmol.types.torch.Tensor[torch.float32][:, :, 3]


   .. py:attribute:: block_coord_offset
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: block_coord_offset64
      :type:  tmol.types.torch.Tensor[torch.int64][:, :]


   .. py:attribute:: inter_residue_connections
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :, 2]


   .. py:attribute:: inter_residue_connections64
      :type:  tmol.types.torch.Tensor[torch.int64][:, :, :, 2]


   .. py:attribute:: inter_block_bondsep
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :, :, :]


   .. py:attribute:: inter_block_bondsep64
      :type:  tmol.types.torch.Tensor[torch.int64][:, :, :, :, :]


   .. py:attribute:: block_type_ind
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: block_type_ind64
      :type:  tmol.types.torch.Tensor[torch.int64][:, :]


   .. py:attribute:: chain_id
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: chain_id64
      :type:  tmol.types.torch.Tensor[torch.int64][:, :]


   .. py:attribute:: pdb_info
      :type:  tmol.pose.pdb_info.PDBInfo


   .. py:attribute:: constraint_set
      :type:  Optional[tmol.pose.constraint_set.ConstraintSet]


   .. py:attribute:: device
      :type:  torch.device


   .. py:attribute:: fragmented_ligand_mapping
      :type:  Optional[tmol.ligand.fragmentation.FragmentedLigandPoseMapping]
      :value: None



   .. py:property:: n_poses


   .. py:property:: max_n_blocks


   .. py:property:: max_n_atoms


   .. py:property:: max_n_block_atoms

      .. rubric:: Docstring

      .. code-block:: text

         Same thing as max_n_atoms


   .. py:property:: max_n_pose_atoms

      .. rubric:: Docstring

      .. code-block:: text

         The largest number of atoms in any pose


   .. py:property:: n_ats_per_block
      :type: tmol.types.torch.Tensor[torch.int64][:, :]


      .. rubric:: Docstring

      .. code-block:: text

         Return the number of atoms in each block


   .. py:property:: real_atoms

      .. rubric:: Docstring

      .. code-block:: text

         return the boolean vector of the real atoms in the coords tensor


   .. py:method:: clone() -> PoseStack

      .. rubric:: Docstring

      .. code-block:: text

         Deep-copy clone of this PoseStack
         


   .. py:method:: split(index) -> PoseStack

      .. rubric:: Docstring

      .. code-block:: text

         Return a single PoseStack from one containing many
         


   .. py:method:: expand_coords()

      .. rubric:: Docstring

      .. code-block:: text

         Load the coordinates into a 4D tensor:
         n_poses x max_n_blocks x max_n_atoms_per_block x 3
         making it possible to perform simple operations on the
         per-block level in python/torch
         


   .. py:property:: n_res_per_pose


   .. py:method:: is_real_block(pose_ind: int, block_ind: int) -> bool

      .. rubric:: Docstring

      .. code-block:: text

         Report whether a particular block on a particular pose is
         real or just filler
         


   .. py:method:: block_type(pose_ind: int, block_ind: int) -> tmol.chemical.restypes.RefinedResidueType

      .. rubric:: Docstring

      .. code-block:: text

         Look up the block type for a particular pose and block and retrieve it
         from the PackedBlockTypes object. is_real_block must return True
         


   .. py:method:: get_constraint_set()


   .. py:method:: block_identity_map()


