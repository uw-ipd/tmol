Score Terms
===========

TMol score functions combine weighted :class:`tmol.ScoreType`
values. A :class:`tmol.ScoreFunction` activates the energy
term implementation associated with each non-zero weight, then renders a
whole-pose, block-pair, or rotamer scoring module for a particular
:class:`tmol.PoseStack`.

Score-type map
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Score types
     - Energy-term class
     - Quantity represented
   * - ``fa_ljatr``, ``fa_ljrep``, ``fa_lk``
     - :class:`~tmol.score.LJLKEnergyTerm`
     - Attractive and repulsive Lennard-Jones interactions and isotropic
       Lazaridis-Karplus solvation.
   * - ``fa_elec``
     - :class:`~tmol.score.ElecEnergyTerm`
     - Distance-dependent full-atom electrostatics.
   * - ``hbond``
     - :class:`~tmol.score.HBondEnergyTerm`
     - Orientation-dependent hydrogen-bond interactions.
   * - ``cart_lengths``, ``cart_angles``, ``cart_torsions``,
       ``cart_impropers``, ``cart_hxltorsions``
     - :class:`~tmol.score.CartBondedEnergyTerm`
     - Bonded geometry used by Cartesian scoring and minimization.
   * - ``constraint``
     - :class:`~tmol.score.ConstraintEnergyTerm`
     - Harmonic distance, bounded distance, harmonic coordinate, and
       circular-harmonic torsion constraints attached through
       :class:`tmol.ConstraintSet`.
   * - ``disulfide``
     - :class:`~tmol.score.DisulfideEnergyTerm`
     - Geometry of disulfide-linked cysteine pairs.
   * - ``omega``, ``rama``
     - :class:`~tmol.score.BackboneTorsionEnergyTerm`
     - Backbone-dependent peptide omega and Ramachandran torsion preferences.
   * - ``dunbrack_rot``, ``dunbrack_rotdev``, ``dunbrack_semirot``
     - :class:`~tmol.score.DunbrackEnergyTerm`
     - Backbone-dependent amino-acid rotamer probabilities and deviations.
   * - ``lk_ball_iso``, ``lk_ball``, ``lk_bridge``, ``lk_bridge_uncpl``
     - :class:`~tmol.score.LKBallEnergyTerm`
     - Directional water-mediated solvation and bridging terms.
   * - ``ref``
     - :class:`~tmol.score.RefEnergyTerm`
     - Per-residue-type reference energies.
   * - ``gen_torsions``
     - :class:`~tmol.score.GenBondedEnergyTerm`
     - Generic bonded torsions, including ligand torsional parameters.
   * - ``na_torsion``, ``na_torsion_well``
     - :class:`~tmol.score.NaTorsionEnergyTerm`
     - DNA/RNA backbone, glycosidic-chi, sugar, coupling, and rotamer-well
       preferences with polymer-specific parameters.

``ScoreType.n_score_types`` is a terminal size sentinel used for weight-vector
allocation; it is not a score component.

Non-bonded and solvation terms
------------------------------

.. autoclass:: tmol.score.LJLKEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.ElecEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.HBondEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.LKBallEnergyTerm
   :show-inheritance:

Bonded, torsional, and reference terms
--------------------------------------

.. autoclass:: tmol.score.CartBondedEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.GenBondedEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.DisulfideEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.BackboneTorsionEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.DunbrackEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.RefEnergyTerm
   :show-inheritance:

Nucleic-acid term
-----------------

.. autoclass:: tmol.score.NaTorsionEnergyTerm
   :show-inheritance:

The :mod:`tmol.score.na_torsion` package exports the sugar-pucker helpers and
fitted-model constants below for lower-level analysis and extension work.

.. automodule:: tmol.score.na_torsion
   :members:
   :imported-members:
   :show-inheritance:

Constraints
-----------

.. autoclass:: tmol.score.ConstraintEnergyTerm
   :show-inheritance:

.. autofunction:: tmol.score.constrain_all_ca

.. autofunction:: tmol.score.create_mainchain_coordinate_constraints

Score-term construction
-----------------------

:class:`tmol.score.ScoreTermFactory` constructs the energy term associated with
each score type. Most applications should configure a
:class:`tmol.ScoreFunction` rather than use the factory directly.

.. autoclass:: tmol.score.ScoreTermFactory
   :members:
   :show-inheritance:
