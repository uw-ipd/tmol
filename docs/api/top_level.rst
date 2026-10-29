Top-level API
=============

The most common TMol objects and workflows are imported directly from the
top-level package, for example ``from tmol import PoseStack``. They are
documented in full on this page. More specialized functionality is exported by
the sub-packages documented in the other API pages, such as :mod:`tmol.io`,
:mod:`tmol.pose`, and :mod:`tmol.score`.

Core objects
------------

.. currentmodule:: tmol

.. autosummary::
   :nosignatures:

   CanonicalOrdering
   CartesianMoveMap
   ConstraintSet
   EdgeType
   FoldForest
   LigandFragmentDefinition
   MoveMap
   PDBInfo
   PackedBlockTypes
   PackerPalette
   PackerTask
   ParameterDatabase
   PatchedChemicalDatabase
   PoseStack
   PoseStackBuilder
   ScoreFunction
   ScoreType

Structure input, output, and visualization
------------------------------------------

.. autosummary::
   :nosignatures:

   atom_array_from_file
   atom_array_from_pose_stack
   extended_pose_stack_from_sequences
   file_from_atom_array
   file_from_pose_stack
   overlay_view
   pose_stack_from_atom_array
   pose_stack_from_file
   selection_gallery
   switchable_view
   view

Pose queries and editing
------------------------

.. autosummary::
   :nosignatures:

   block_for_residue_label
   get_ligand_mask
   get_sequence
   get_torsion_names
   mutate_residue
   mutate_to_sequence
   recombine_fragmented_ligands

Scoring, packing, and optimization
----------------------------------

.. autosummary::
   :nosignatures:

   beta2016_score_function
   calculate_fragment_interactions
   fast_relax
   pack_rotamers
   run_cart_min
   run_kin_min

Environment
-----------

.. autosummary::
   :nosignatures:

   has_cuda

Reference
---------

.. automodule:: tmol
   :members:
   :imported-members:
   :show-inheritance:
