Relax
=====

``cartesian_fast_relax`` and ``kin_fast_relax`` alternate side-chain packing
with their respective minimizers while ramping repulsive and optional
constraint weights. ``fast_relax`` is the lower-level configurable protocol.
Use :class:`tmol.kinematics.move_map.CartesianMoveMap` to select movable
Cartesian coordinates or :class:`tmol.kinematics.move_map.MoveMap` with a
:class:`tmol.kinematics.fold_forest.FoldForest` for kinematic minimization.

.. automodule:: tmol.relax.fast_relax
   :members:
   :undoc-members:
   :show-inheritance:
