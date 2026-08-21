Score Terms
===========

TMol score functions combine weighted :class:`tmol.score.ScoreType`
values. A :class:`tmol.score.ScoreFunction` activates the energy
term implementation associated with each non-zero weight, then renders a
whole-pose, block-pair, or rotamer scoring module for a particular
:class:`tmol.pose.PoseStack`.

Score-type map
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Score types
     - Energy-term class
     - Quantity represented
   * - ``fa_ljatr``, ``fa_ljrep``, ``fa_lk``
     - :class:`~tmol.score.ljlk.LJLKEnergyTerm`
     - Attractive and repulsive Lennard-Jones interactions and isotropic
       Lazaridis-Karplus solvation.
   * - ``fa_elec``
     - :class:`~tmol.score.elec.ElecEnergyTerm`
     - Distance-dependent full-atom electrostatics.
   * - ``hbond``
     - :class:`~tmol.score.hbond.HBondEnergyTerm`
     - Orientation-dependent hydrogen-bond interactions.
   * - ``cart_lengths``, ``cart_angles``, ``cart_torsions``,
       ``cart_impropers``, ``cart_hxltorsions``
     - :class:`~tmol.score.cartbonded.CartBondedEnergyTerm`
     - Bonded geometry used by Cartesian scoring and minimization.
   * - ``constraint``
     - :class:`~tmol.score.constraint.ConstraintEnergyTerm`
     - Harmonic distance, bounded distance, harmonic coordinate, and
       circular-harmonic torsion constraints attached through
       :class:`tmol.pose.ConstraintSet`.
   * - ``disulfide``
     - :class:`~tmol.score.disulfide.DisulfideEnergyTerm`
     - Geometry of disulfide-linked cysteine pairs.
   * - ``omega``, ``rama``
     - :class:`~tmol.score.backbone_torsion.BackboneTorsionEnergyTerm`
     - Backbone-dependent peptide omega and Ramachandran torsion preferences.
   * - ``dunbrack_rot``, ``dunbrack_rotdev``, ``dunbrack_semirot``
     - :class:`~tmol.score.dunbrack.DunbrackEnergyTerm`
     - Backbone-dependent amino-acid rotamer probabilities and deviations.
   * - ``lk_ball_iso``, ``lk_ball``, ``lk_bridge``, ``lk_bridge_uncpl``
     - :class:`~tmol.score.lk_ball.LKBallEnergyTerm`
     - Directional water-mediated solvation and bridging terms.
   * - ``ref``
     - :class:`~tmol.score.ref.RefEnergyTerm`
     - Per-residue-type reference energies.
   * - ``gen_torsions``
     - :class:`~tmol.score.genbonded.GenBondedEnergyTerm`
     - Generic bonded torsions, including ligand torsional parameters.
   * - ``na_torsion``, ``na_torsion_well``
     - :class:`~tmol.score.na_torsion.NaTorsionEnergyTerm`
     - DNA/RNA backbone, glycosidic-chi, sugar, coupling, and rotamer-well
       preferences with polymer-specific parameters.

``ScoreType.n_score_types`` is a terminal size sentinel used for weight-vector
allocation; it is not a score component.

Non-bonded and solvation terms
------------------------------

.. autoclass:: tmol.score.ljlk.LJLKEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.elec.ElecEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.hbond.HBondEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.lk_ball.LKBallEnergyTerm
   :show-inheritance:

Bonded, torsional, and reference terms
--------------------------------------

.. autoclass:: tmol.score.cartbonded.CartBondedEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.genbonded.GenBondedEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.disulfide.DisulfideEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.backbone_torsion.BackboneTorsionEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.dunbrack.DunbrackEnergyTerm
   :show-inheritance:

.. autoclass:: tmol.score.ref.RefEnergyTerm
   :show-inheritance:

Nucleic-acid term
-----------------

.. autoclass:: tmol.score.na_torsion.NaTorsionEnergyTerm
   :show-inheritance:

.. automodule:: tmol.score.na_torsion
   :show-inheritance:

Constraints
-----------

.. autoclass:: tmol.score.constraint.ConstraintEnergyTerm
   :show-inheritance:

.. automodule:: tmol.score.constraint
   :show-inheritance:
