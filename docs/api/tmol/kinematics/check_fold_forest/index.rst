tmol.kinematics.check_fold_forest
=================================

.. py:module:: tmol.kinematics.check_fold_forest


Functions
---------

.. autoapisummary::

   tmol.kinematics.check_fold_forest.mark_polymeric_bonds_in_foldforest_edges
   tmol.kinematics.check_fold_forest.bfs_proper_forest
   tmol.kinematics.check_fold_forest.ensure_jumps_numbered_and_distinct
   tmol.kinematics.check_fold_forest.validate_fold_forest_jit
   tmol.kinematics.check_fold_forest.validate_fold_forest


Module Contents
---------------

.. py:function:: mark_polymeric_bonds_in_foldforest_edges(n_poses: int, max_n_blocks: int, n_blocks: tmol.types.array.NDArray[int][:], edges: tmol.types.array.NDArray[int][:, :, 4])

   .. rubric:: Docstring

   .. code-block:: text

      Make each implicit i-to-i+1 or i-to-(i-1) polymer bond explicit
      
      .. rubric:: Notes
      
      This code does not ensure that the polymeric bonds between
      these two residues are present in the PoseStack; this means
      that if there are missing loops, e.g., that we can still
      "fold through" them.
      

.. py:function:: bfs_proper_forest(roots: tmol.types.array.NDArray[numpy.int64][:, :], n_blocks: tmol.types.array.NDArray[numpy.int64][:], connections: tmol.types.array.NDArray[numpy.int64][:, :, :])

.. py:function:: ensure_jumps_numbered_and_distinct(edges: tmol.types.array.NDArray[numpy.int64][:, :, 4])

.. py:function:: validate_fold_forest_jit(n_blocks: tmol.types.array.NDArray[numpy.int64][:], edges: tmol.types.array.NDArray[numpy.int64][:, :, 4])

.. py:function:: validate_fold_forest(n_blocks: tmol.types.array.NDArray[numpy.int64][:], edges: tmol.types.array.NDArray[numpy.int64][:, :, 4])

