tmol.io.chain_deduction
=======================

.. py:module:: tmol.io.chain_deduction


Functions
---------

.. autoapisummary::

   tmol.io.chain_deduction.chain_inds_for_pose_stack
   tmol.io.chain_deduction.annotate_pbt_w_valid_connection_masks


Module Contents
---------------

.. py:function:: chain_inds_for_pose_stack(pose_stack: tmol.pose.pose_stack.PoseStack) -> tmol.types.array.NDArray[numpy.int64][:, :]

   .. rubric:: Docstring

   .. code-block:: text

      Label each residue by which chain it comes from, where "chain" is a group
      of polymer residues that are connected by certain sets of bonds (e.g. not
      disulfide bonds). This problem becomes one of finding the connected components
      of a graph and is handled using scipy's (CPU) code. Gap residues are given a
      chain ID of -1.
      

.. py:function:: annotate_pbt_w_valid_connection_masks(pbt: tmol.pose.packed_block_types.PackedBlockTypes)

   .. rubric:: Docstring

   .. code-block:: text

      We want to take the up-down polymeric connections between residues
      that have up-down connections and not other connections, unless
      otherwise instructed.
      
      The logic here is to take the up- and down-connections from
      polymeric residues as the ones that connect two residues part
      of the same chain. This would make the C->N connection along
      a protein backbone serve to say residues i and i+1 are part
      of the same chain without saying that a disulfide bond
      between residues i and j make them part of the same chain.
      (They are at that point a single molecule, but, conceptually
      still separate chains.)
      
      For non-polymeric residues, all their chemical bonds should
      be considered as connecting them to members of their same chain.
      
      The upshot is: if a polymeric residue is connected to a
      non-polymeric residue through one of its non-up/non-down
      connection points, the non-polymeric residue will still be
      considered part of the polymeric residue's chain. Either
      connection direction is sufficient to link two residues
      as part of the same chain.
      

