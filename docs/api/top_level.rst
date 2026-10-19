Top-level API
=============

The top-level package provides convenience imports for the most common TMol
objects and workflows. Their full documentation lives in the owning packages,
such as :mod:`tmol.io`, :mod:`tmol.pose`, and :mod:`tmol.score`.

Core objects
------------

.. currentmodule:: tmol

.. autosummary::
   :nosignatures:

   CanonicalOrdering
   CartesianMoveMap
   ConstraintEnergyTerm
   ConstraintSet
   EdgeType
   FoldForest
   KinematicModuleData
   MoveMap
   PackedBlockTypes
   ParameterDatabase
   PoseStack
   ScoreFunction
   ScoreType

Structure and chemistry
-----------------------

.. autosummary::
   :nosignatures:

   atom_array_from_file
   atom_array_from_pose_stack
   default_canonical_ordering
   default_packed_block_types
   extended_pose_stack_from_sequences
   file_from_atom_array
   file_from_pose_stack
   one2three
   pose_stack_from_atom_array
   pose_stack_from_canonical_form
   pose_stack_from_file
   selection_gallery
   switchable_view
   three2one
   view

Scoring and optimization
------------------------

.. autosummary::
   :nosignatures:

   beta2016_score_function
   build_kinforest_network
   create_mainchain_coordinate_constraints
   fast_relax
   get_named_torsions
   get_torsion_names
   run_cart_min
   run_kin_min
   run_min
   set_named_torsions

C++ and CUDA integration
------------------------

.. automodule:: tmol
   :members: include_paths
   :show-inheritance:
