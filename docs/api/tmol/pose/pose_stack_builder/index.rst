tmol.pose.pose_stack_builder
============================

.. py:module:: tmol.pose.pose_stack_builder


Classes
-------

.. autoapisummary::

   tmol.pose.pose_stack_builder.PoseStackBuilder


Module Contents
---------------

.. py:class:: PoseStackBuilder

   .. py:method:: from_poses(pose_stacks: List[tmol.pose.pose_stack.PoseStack], device: torch.device) -> tmol.pose.pose_stack.PoseStack
      :classmethod:



   .. py:method:: pose_stack_from_monomer_polymer_sequences(packed_block_types: tmol.pose.packed_block_types.PackedBlockTypes, sequences)
      :classmethod:



   .. py:method:: pose_stack_from_monomer_sequences_w_extrapolymeric_conns(packed_block_types: tmol.pose.packed_block_types.PackedBlockTypes, sequences)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Construct a PoseStack given a list of sequences where the disulfide
         connectivity is known. E.g. If there is a disulfide pair between residues
         5 and 20 and another disulfide pair between residues 9 and 15, then
         the sequence would be given as:
         
         AAAA[CYD--dslf-first]AAA[CYD--dslf-second]AAA ...
         AA[CYD--dslf-second]AAAA[CYD--dslf-first]AAA
         
         where the string following the double dash, designates 1) the name of
         the inter-residue connection (for CYD, this is "dslf") and then 2) after
         the single dash, a unique identifier so that which pair of residues are
         forming that connection. In this case the two disulfides have the labels
         "first" and "second," but any unique label would suffice.
         
         


   .. py:method:: pose_stack_from_sequences(packed_block_types: tmol.pose.packed_block_types.PackedBlockTypes, sequences, chain_lengths)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Construct a PoseStack given a list of sequences where the disulfide
         connectivity is known. E.g. If there is a disulfide pair between
         residues 5 and 20 and another disulfide pair between residues 9 and 15,
         then the sequence would be given as:
         
         AAAA[CYD--dslf-first]AAA[CYD--dslf-second]AAA ...
         AA[CYD--dslf-second]AAAA[CYD--dslf-first]AAA
         
         where the string following the double dash, designates 1) the name of the
         inter-residue connection (for CYD, this is "dslf") and then 2) after the
         single dash, a unique identifier so that which pair of residues are forming
         that connection. In this case the two disulfides have the labels "first"
         and "second," but any unique label would suffice.
         
         


   .. py:method:: rebuild_with_new_packed_block_types(ps: tmol.pose.pose_stack.PoseStack, packed_block_types: tmol.pose.packed_block_types.PackedBlockTypes)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Create a new PoseStack object replacing the existing PackedBlockTypes
         object with a new one, and then rebuilding the other data members that
         depend on it.
         


