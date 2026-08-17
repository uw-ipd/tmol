Score Terms
===========

TMol score functions combine weighted :class:`tmol.score.score_types.ScoreType`
values. A :class:`tmol.score.score_function.ScoreFunction` activates the energy
term implementation associated with each non-zero weight, then renders a
whole-pose, block-pair, or rotamer scoring module for a particular
:class:`tmol.pose.pose_stack.PoseStack`.

Score-type map
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 28 48

   * - Score types
     - Energy-term class
     - Quantity represented
   * - ``fa_ljatr``, ``fa_ljrep``, ``fa_lk``
     - :class:`~tmol.score.ljlk.ljlk_energy_term.LJLKEnergyTerm`
     - Attractive and repulsive Lennard-Jones interactions and isotropic
       Lazaridis-Karplus solvation.
   * - ``fa_elec``
     - :class:`~tmol.score.elec.elec_energy_term.ElecEnergyTerm`
     - Distance-dependent full-atom electrostatics.
   * - ``hbond``
     - :class:`~tmol.score.hbond.hbond_energy_term.HBondEnergyTerm`
     - Orientation-dependent hydrogen-bond interactions.
   * - ``cart_lengths``, ``cart_angles``, ``cart_torsions``,
       ``cart_impropers``, ``cart_hxltorsions``
     - :class:`~tmol.score.cartbonded.cartbonded_energy_term.CartBondedEnergyTerm`
     - Bonded geometry used by Cartesian scoring and minimization.
   * - ``constraint``
     - :class:`~tmol.score.constraint.constraint_energy_term.ConstraintEnergyTerm`
     - Harmonic distance, bounded distance, harmonic coordinate, and
       circular-harmonic torsion constraints attached through
       :class:`tmol.pose.constraint_set.ConstraintSet`.
   * - ``disulfide``
     - :class:`~tmol.score.disulfide.disulfide_energy_term.DisulfideEnergyTerm`
     - Geometry of disulfide-linked cysteine pairs.
   * - ``omega``, ``rama``
     - :class:`~tmol.score.backbone_torsion.bb_torsion_energy_term.BackboneTorsionEnergyTerm`
     - Backbone-dependent peptide omega and Ramachandran torsion preferences.
   * - ``dunbrack_rot``, ``dunbrack_rotdev``, ``dunbrack_semirot``
     - :class:`~tmol.score.dunbrack.dunbrack_energy_term.DunbrackEnergyTerm`
     - Backbone-dependent amino-acid rotamer probabilities and deviations.
   * - ``lk_ball_iso``, ``lk_ball``, ``lk_bridge``, ``lk_bridge_uncpl``
     - :class:`~tmol.score.lk_ball.lk_ball_energy_term.LKBallEnergyTerm`
     - Directional water-mediated solvation and bridging terms.
   * - ``ref``
     - :class:`~tmol.score.ref.ref_energy_term.RefEnergyTerm`
     - Per-residue-type reference energies.
   * - ``gen_torsions``
     - :class:`~tmol.score.genbonded.genbonded_energy_term.GenBondedEnergyTerm`
     - Generic bonded torsions, including ligand torsional parameters.
   * - ``na_torsion``, ``na_torsion_well``
     - :class:`~tmol.score.na_torsion.na_torsion_energy_term.NaTorsionEnergyTerm`
     - DNA/RNA backbone, glycosidic-chi, sugar, coupling, and rotamer-well
       preferences with polymer-specific parameters.

``ScoreType.n_score_types`` is a terminal size sentinel used for weight-vector
allocation; it is not a score component.

Non-bonded and solvation terms
------------------------------

.. autoclass:: tmol.score.ljlk.ljlk_energy_term.LJLKEnergyTerm
   :members:
   :show-inheritance:

.. autoclass:: tmol.score.elec.elec_energy_term.ElecEnergyTerm
   :members:
   :show-inheritance:

.. autoclass:: tmol.score.hbond.hbond_energy_term.HBondEnergyTerm
   :members:
   :show-inheritance:

.. autoclass:: tmol.score.lk_ball.lk_ball_energy_term.LKBallEnergyTerm
   :members:
   :show-inheritance:

Bonded, torsional, and reference terms
--------------------------------------

.. autoclass:: tmol.score.cartbonded.cartbonded_energy_term.CartBondedEnergyTerm
   :members:
   :show-inheritance:

.. autoclass:: tmol.score.genbonded.genbonded_energy_term.GenBondedEnergyTerm
   :members:
   :show-inheritance:

.. autoclass:: tmol.score.disulfide.disulfide_energy_term.DisulfideEnergyTerm
   :members:
   :show-inheritance:

.. autoclass:: tmol.score.backbone_torsion.bb_torsion_energy_term.BackboneTorsionEnergyTerm
   :members:
   :show-inheritance:

.. autoclass:: tmol.score.dunbrack.dunbrack_energy_term.DunbrackEnergyTerm
   :members:
   :show-inheritance:

.. autoclass:: tmol.score.ref.ref_energy_term.RefEnergyTerm
   :members:
   :show-inheritance:

Nucleic-acid term
-----------------

.. autoclass:: tmol.score.na_torsion.na_torsion_energy_term.NaTorsionEnergyTerm
   :members:
   :show-inheritance:

.. automodule:: tmol.score.na_torsion.params
   :members:
   :show-inheritance:

Constraints
-----------

.. autoclass:: tmol.score.constraint.constraint_energy_term.ConstraintEnergyTerm
   :members:
   :show-inheritance:

.. automodule:: tmol.score.constraint.utility
   :members:
   :show-inheritance:
