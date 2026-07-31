tmol.ligand.chi_topology
========================

.. py:module:: tmol.ligand.chi_topology

.. rubric:: Module docstring

.. code-block:: text

   Rotatable-bond (CHI / PROTON_CHI) classification for ligand residue types.
   
   Ports RosettaVS ``generic_potential`` ``define_rotable_torsions``
   (``SetupTopology.py``) and the PROTON_CHI sample tables (``Molecule.py``)
   into tmol's ligand-preparation pipeline.  The goal is *semantic* parity with
   RosettaVS: the same set of rotatable bond axes plus correct proton-chi sample
   sets, not byte-identical CHI numbering.
   
   The classifier is pure: it consumes the RDKit ``Mol``, the deterministic
   atom-tree already built by :mod:`tmol.ligand.residue_builder`
   (``order``/``parent``/``grandparents``), the per-atom names, and the
   ``RosettaTypingState`` produced by :func:`tmol.ligand.atom_typing.assign_tmol_atom_types`
   (``return_state=True``).  It emits named ``Torsion`` objects (``chi1``..``chiN``)
   and, for polar-hydrogen rotations, matching ``ChiSamples``.
   
   Hard-coded RosettaVS default flags (see ``BasicClasses.py``):
   ``report_Hapol_chi=False``, ``report_amide_chi=False``,
   ``report_nbonded_chi=False``, ``report_ringring_chi=True``,
   ``report_puckering_chi=False``, ``max_confs=5000``.
   
   Validated against the RosettaVS ground truth (``ref1``/``ref2`` via the SMILES
   path in ``TestGroundTruthRegression``): emitted CHI axes and PROTON_CHI
   samples/expansions match. The ``EXTRA`` encoding (``EXTRA 1 20`` ->
   ``expansions=(20.0,)``, ``EXTRA 0`` -> ``()``) is consistent with
   ``OptHSampler``'s ``len(samples) * (1 + 2 * len(expansions))`` expansion.
   
   Scope notes / latitude:
   - Conjugated-polar-H skipping is a faithful port of ``assign_bond_conjugation``'s
     core: a bond is conjugated only when both atom classes are in
     :data:`_CONJUGATING_ACLASSES` (and neither is sp3), plus Rosetta's all-but-one-H
     test. So phenol/acid C-OH and aniline/amide -NH are classified like
     mol2genparams (verified). The geometry-based planarity refinement (``is_planar``)
     is not ported.
   - ``border > 1`` biaryl-pivot CHIs (ring <-> conjugated functional group) ARE
     emitted via a port of ``search_special_biaryl_ring`` (the hard-coded
     :data:`_SPECIAL_BIARYL_PAIRS` class-pair list). The remaining pivots through
     ring-like conjugated groups (guanidinium ``Ngu1``, tertiary amide ``Nad3``,
     ``NG2``, furan ``Ofu``) are recovered not by the ``is_planar`` geometry port
     but by honoring the source mol2's literal single-bond order
     (``original_single_bonds``): RDKit kekulization promotes those aryl-X bonds to
     DOUBLE, which the ``border > 1`` rule would skip, whereas Rosetta reads the
     mol2 order ``1`` verbatim. Restoring ``border = 1`` for those bonds matches
     mol2genparams and closes the DUD-80 parity set (80/80 full CHI).
   - NU / ring-pucker DOFs are unsupported (RosettaVS default
     ``report_puckering_chi=False``); none are emitted by any preparation path.
   
   RosettaVS rules that are handled *implicitly* by emitting CHIs only for
   atom-tree edges (each non-root atom ``c`` with parent ``b``), rather than over
   a separately-enumerated torsion list:
   - ``ring_cuts``: a ring's closure bond is a non-tree (back) edge, so it is never
     a parent->child edge and never a CHI candidate.
   - ``FT_connected``: fold-tree-disconnected torsions cannot arise — every emitted
     axis is a tree edge by construction.
   - ``atms_puckering`` (default-off): puckering-ring internal bonds are
     ring-internal and are already skipped by ``_share_ring``.
   num_H_confs is computed over the pre-skip polar-H set (so EXTRA matches
   RosettaVS even when a polar-H chi is counted but later skipped).
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.chi_topology.MAX_CONFS


Functions
---------

.. autoapisummary::

   tmol.ligand.chi_topology.build_chi_topology


Module Contents
---------------

.. py:data:: MAX_CONFS
   :value: 5000


.. py:function:: build_chi_topology(mol: rdkit.Chem.Mol, order: list[int], parent: dict[int, int], grandparents: dict[int, tuple[int, int]], atom_names: list, typing_state, *, atype_by_idx: dict[int, str] | None = None, original_single_bonds: frozenset[frozenset[str]] | None = None, logger=None) -> tuple[tuple[tmol.database.chemical.Torsion, Ellipsis], tuple[tmol.database.chemical.ChiSamples, Ellipsis]]

   .. rubric:: Docstring

   .. code-block:: text

      Classify rotatable bonds and return ``(torsions, chi_samples)``.
      
      ``order``/``parent``/``grandparents`` are the kept-atom tree from
      ``build_residue_type`` (indices are RDKit atom indices; ``parent[root]``
      is the root itself).  ``atom_names[idx]`` is the final residue atom name
      (or ``None`` for dropped atoms).  ``typing_state`` is a
      :class:`~tmol.ligand.atom_typing.RosettaTypingState`.
      
      ``original_single_bonds`` (optional) is a set of ``frozenset({name_a,
      name_b})`` pairs that the source mol2 records as literal single bonds.
      For those bonds the bond order is forced to 1 — overriding RDKit's
      post-kekulization promotion of some aromatic/conjugated single bonds to
      DOUBLE — so the ``border > 1`` skips match Rosetta's ``mol2genparams``,
      which reads the literal mol2 order.
      

