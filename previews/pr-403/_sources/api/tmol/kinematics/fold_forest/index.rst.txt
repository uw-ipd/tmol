tmol.kinematics.fold_forest
===========================

.. py:module:: tmol.kinematics.fold_forest


Classes
-------

.. autoapisummary::

   tmol.kinematics.fold_forest.EdgeType
   tmol.kinematics.fold_forest.FoldForest


Module Contents
---------------

.. py:class:: EdgeType

   Bases: :py:obj:`enum.IntEnum`


   .. rubric:: Docstring

   .. code-block:: text

      Enum where members are also (and must be) ints
      

   .. py:attribute:: polymer
      :value: 0



   .. py:attribute:: jump


   .. py:attribute:: root_jump


.. py:class:: FoldForest

   .. rubric:: Docstring

   .. code-block:: text

      The fold forest will define the fold trees for the poses in a PoseStack.
      Each tensor in the class has its first dimension over the number of poses.
      
      The primary definition of a FoldTree is the Edge. The Edge defines a connection
      between two parts of a Pose. The three types of edges are 1. polymer edges
      (analgous to the previously named "peptide edges" from Rosetta++ and Rosetta3), 2.
      jump edges which connect any pair of residues in the Pose, and 3. root-jump
      edges, which originate at the explicit virtual root and connect to a particular
      residue. A polymer edge spans a contiguous range of polymeric block types where
      the "up" connection of residue i is connected to the "down" connection of residue
      i+1 for all i in the range between the "start" and "end" blocks.
      
      Each edge is described by a 4-tuple of integers (type, start, end, jump-index);
      where type is one of the EdgeType enum values, start is the index of the upstream
      residue of the edge, end is the index of the downstream residue of the edge, and
      jump-index is used to assign an id to any particular jump edge; jump-edge indices
      must be unique and ascending from 0 to n_jumps-1. "Root jump" edges take their
      "identity" from the downstream residue of the edge, so they do not need an index.
      
      The FoldForest in tmol differs from the FoldTree in Rosetta3 in that there
      is always a virtual root at the origin and any residue (block) may be
      connected to this root by a "root jump". Such root-jump residues are defined
      by listing the residue that the root is connected to as the "end" residue;
      the "start" residue field should be left as -1. An example FoldForest for a
      ten-residue protein might be:
        (polymer, 0, 4)
        (jump   , 0, 7)
        (polymer, 7, 9)
        (polymer, 7, 6)
        (root-jump, -1, 0)
        (root-jump, -1, 5)
      where both residues are 0 and 5 are connected to the root.
      
      Note that in the MoveMap, the root-jumps are distinct from the non-root-jumps.
      

   .. py:attribute:: max_n_edges
      :type:  int


   .. py:attribute:: n_edges
      :type:  tmol.types.array.NDArray[int][:]


   .. py:attribute:: edges
      :type:  tmol.types.array.NDArray[int][:, :, 4]


   .. py:method:: reasonable_fold_forest(pose_stack: tmol.pose.pose_stack.PoseStack)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Create a fold forest for each pose using only backbone (up/down)
         polymer connectivity.
         
         Each biological chain (same chain_id) is rooted with a single
         root-jump to its first residue.  Polymer gaps within that chain
         (chain breaks) become ordinary jump edges connecting the last
         residue before the gap to the first residue after it.  Gaps
         between different biological chains produce separate root-jumps.
         Cyclic polymers (C→N cyclisation) are broken at the bond entering
         the lowest-index residue; that bond is dropped to keep the forest
         a valid tree.  Non-polymer connections (disulfides, etc.) are ignored.
         


   .. py:method:: from_edges(edges: tmol.types.array.NDArray[int][:, :, 4])
      :classmethod:



