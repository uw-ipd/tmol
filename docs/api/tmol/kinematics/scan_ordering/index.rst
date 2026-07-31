tmol.kinematics.scan_ordering
=============================

.. py:module:: tmol.kinematics.scan_ordering


Classes
-------

.. autoapisummary::

   tmol.kinematics.scan_ordering.KinForestScanOrdering
   tmol.kinematics.scan_ordering.ResidueKinforestData


Functions
---------

.. autoapisummary::

   tmol.kinematics.scan_ordering.get_children
   tmol.kinematics.scan_ordering.get_scans
   tmol.kinematics.scan_ordering.construct_kin_module_data_for_pose
   tmol.kinematics.scan_ordering.annotate_block_type_with_residue_kinforest_data


Module Contents
---------------

.. py:function:: get_children(parents)

.. py:function:: get_scans(parents, roots)

   .. rubric:: Docstring

   .. code-block:: text

      Partitioning of a tree into linear subpaths.
      
      The tree is stored as a list of pointers to parent indices, with an
      arbitrary set of roots.  Each root implies a connected subtree.
      Each index i has a parent index (p_i) "higher" in the tree (p_i < i).
      The tree structure is fully defined by parent pointers, an index array
      (parent) of len(tree_size) where parent[i] == i for all root nodes.
      
      The source tree implies a per-node child list of length >= 0, defined by
      all nodes for which the given node is a parent. Nodes with no children are
      leaves.
      
      Scan paths cuts the tree into linear paths, where each non-leaf node (i)
      has a *single* child (c_i) "lower" in the tree (i < c_i). The path
      structure is fully defined by child pointers, an index array (subpath_child)
      of subpath_child[i] == c_i (non-leaves) or subpath_child[i] = -1 (leaves).
      
      The path partitioning implies "subpath roots", the set of nodes which are
      *not* the subpath child of their parent (subpath_child[parent[i]] != i).
      
      Each path in the tree is labeled with a depth: a path with depth
      i may depend on the values computed for atoms with depths 0..i-1.
      All of the paths of the same depth can be processed in a single
      kernel execution with segmented scan.
      

.. py:class:: KinForestScanOrdering

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      Scan plans for parallel kinematic operations.
      
      The KinForestScanOrdering class divides the tree into a set of paths. Along
      each path is a continuous chain of atoms that either (1) require their
      coordinate frames computed as a cumulative product of homogeneous
      transforms for the coordinate update algorithm, or (2) require the
      cumulative sum of their f1f2 vectors for the derivative calculation
      algorithm. In both cases, these paths can be processed efficiently
      on the GPU using an algorithm called "scan" and batches of these paths
      can be processed at once in a variant called "segmented scan."
      
      The same set of paths is used for both the refold algorithm and
      the derivative summation; the refold algorithm starts at path
      roots and multiplies homogeneous transforms towards the leaves.
      The derivative summation algorithm starts at the leaves and sums
      upwards towards the roots.
      
      To accomplish this, the GPUKinForestReordering class reorders the atoms from
      the original KinForest order ("ko") where atoms are known by their
      kinforest-index ("ki") into 1) their refold order ("ro") where atoms are
      known by their refold index ("ri") and 2) their deriv-sum order ("dso")
      where atoms are known by their deriv-sum index.
      
      Each scan operation is performed as a series of depth-based "generations",
      in which the scans at generation n are *only* dependent on the results of
      scans in generations [0...n-1] inclusive. In the forward-scan generations
      are ordered from the kinematic root, and scan segments begin with a value
      derived from the result of a generation [0...n-1] scan. In the
      backward-scan generations are ordered from kinematic leaves, and scan
      segments pull summation results from multiple generation [0...n-1] scans.
      
      For further details on parallel segmented scan operations see:
      
      * Mark Harris, "Parallel Prefix Sum with CUDA."
        GPU Gems 3, Nvidia Corporation
        https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html
        http://developer.download.nvidia.com/
        compute/cuda/2_2/sdk/website/projects/scan/doc/scan.pdf
      
      * Sengupta, Shubhabrata, et al. "Scan primitives for GPU computing."
        Graphics hardware. Vol. 2007. 2007.
        http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Sengupta07.pdf
      
      * Sean Baxter, "moderngpu"
        http://moderngpu.github.io/moderngpu
      

   .. py:attribute:: kinforest_cache_key
      :value: '__KinForestScanOrdering_cache__'



   .. py:attribute:: forward_scan_paths
      :type:  tmol.kinematics.datatypes.KinForestScanData


   .. py:attribute:: backward_scan_paths
      :type:  tmol.kinematics.datatypes.KinForestScanData


   .. py:method:: for_kinforest(kinforest)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Calculate and cache refold ordering over kinforest
         
         KinForest data structure is frozen; so it is safe to cache the gpu scan
         ordering for a single object. Store as a private property of the input
         kinforest, lifetime of the cache will then be managed via the target
         object.
         .
         


   .. py:method:: calculate_from_kinforest(kinforest: tmol.kinematics.datatypes.KinForest)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Setup for operations over KinForest.
         ``device`` is inferred from kinforest tensor device.
         


.. py:function:: construct_kin_module_data_for_pose(pose_stack: tmol.pose.pose_stack.PoseStack, fold_forest_edges: tmol.types.torch.Tensor[torch.int32][:, :, 4])

.. py:class:: ResidueKinforestData

   .. rubric:: Docstring

   .. code-block:: text

      Per-block-type kinforest data rooted at the default jump connection
      atom (e.g. CA for amino acids, O for water).
      
      Produced by annotate_block_type_with_residue_kinforest_data and consumed
      by construct_single_residue_kinforest to build the rotamer kinforest
      without the deprecated _KinematicBuilder.
      

   .. py:attribute:: jump_atom
      :type:  int


   .. py:attribute:: preds
      :type:  tmol.types.array.NDArray[numpy.int64][:]


   .. py:attribute:: bfto_2_orig
      :type:  tmol.types.array.NDArray[numpy.int64][:]


   .. py:attribute:: dof_type
      :type:  tmol.types.array.NDArray[numpy.int64][:]


.. py:function:: annotate_block_type_with_residue_kinforest_data(bt)

   .. rubric:: Docstring

   .. code-block:: text

      Annotate a block type with the single-residue kinforest spanning tree,
      rooted at its default jump connection atom (CA for amino acids, O for water).
      
      Caches:
        bt.residue_kinforest_data  -- ResidueKinforestData (public, used by rotamer packing)
        bt._bond_graph_mst         -- scipy MST (private, reused by
                                      _annotate_block_type_with_gen_scan_path_segs)
      

