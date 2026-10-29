Pose
====

:class:`tmol.PoseStack` stores a padded batch of molecular coordinates and
block topology. :class:`tmol.PoseStackBuilder` combines compatible poses,
while :class:`tmol.PackedBlockTypes` holds the shared block-type vocabulary.
Constraint data is stored in :class:`tmol.ConstraintSet`. These classes, and
the pose queries :func:`tmol.get_sequence`, :func:`tmol.get_ligand_mask`,
:func:`tmol.block_for_residue_label`, and :func:`tmol.get_torsion_names`, are
imported from the top-level package and documented in :doc:`top_level`.

The :mod:`tmol.pose` package provides the labelling and selection helpers
below.

.. automodule:: tmol.pose
   :members:
   :imported-members:
   :show-inheritance:
