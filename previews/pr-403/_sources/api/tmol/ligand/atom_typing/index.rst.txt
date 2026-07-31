tmol.ligand.atom_typing
=======================

.. py:module:: tmol.ligand.atom_typing

.. rubric:: Module docstring

.. code-block:: text

   Atom type assignment for ligand atoms.
   
   Assigns Rosetta generic_potential atom types to atoms in an RDKit Mol.
   The classification logic is a faithful port of Rosetta's AtomTypeClassifier
   (from mol2genparams / generic_potential) and produces identical atom types
   and atom names, including the polar-carbon modifier and the Rosetta hydrogen
   naming convention (H<bonded_element><count>).
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.atom_typing.logger
   tmol.ligand.atom_typing.HYB_SP
   tmol.ligand.atom_typing.HYB_SP2
   tmol.ligand.atom_typing.HYB_SP3
   tmol.ligand.atom_typing.HYB_AMIDE
   tmol.ligand.atom_typing.HYB_AROMATIC
   tmol.ligand.atom_typing.ELEMENT_SYMBOLS


Classes
-------

.. autoapisummary::

   tmol.ligand.atom_typing.AtomTypeAssignment
   tmol.ligand.atom_typing.RosettaTypingState


Functions
---------

.. autoapisummary::

   tmol.ligand.atom_typing.sanitize_tolerant
   tmol.ligand.atom_typing.kekulize_tolerant
   tmol.ligand.atom_typing.should_kekulize_for_typing
   tmol.ligand.atom_typing.assign_tmol_atom_types


Module Contents
---------------

.. py:data:: logger

.. py:data:: HYB_SP
   :value: 1


.. py:data:: HYB_SP2
   :value: 2


.. py:data:: HYB_SP3
   :value: 3


.. py:data:: HYB_AMIDE
   :value: 8


.. py:data:: HYB_AROMATIC
   :value: 9


.. py:data:: ELEMENT_SYMBOLS

.. py:class:: AtomTypeAssignment

   Bases: :py:obj:`NamedTuple`


   .. rubric:: Docstring

   .. code-block:: text

      Atom-type result for a single atom: name, type, element, and index.
      

   .. py:attribute:: atom_name
      :type:  str


   .. py:attribute:: atom_type
      :type:  str


   .. py:attribute:: element
      :type:  str


   .. py:attribute:: index
      :type:  int


.. py:class:: RosettaTypingState

   .. rubric:: Docstring

   .. code-block:: text

      Precomputed state consumed by Rosetta-style classifiers.
      

   .. py:attribute:: source_subtype_by_idx
      :type:  dict[int, str]


   .. py:attribute:: hyb_by_idx
      :type:  dict[int, int]


   .. py:attribute:: atms_aro
      :type:  set[int]


   .. py:attribute:: atms_strained
      :type:  set[int]


   .. py:attribute:: rings
      :type:  list[tuple[int, Ellipsis]]
      :value: []



   .. py:attribute:: ring_membership_by_idx
      :type:  dict[int, set[int]]


   .. py:attribute:: neighbor_counts_by_idx
      :type:  dict[int, tuple[int, int, int, int, int, int]]


.. py:function:: sanitize_tolerant(mol: rdkit.Chem.Mol) -> None

   .. rubric:: Docstring

   .. code-block:: text

      Run ``Chem.SanitizeMol`` with a Kekulé/valence-tolerant fallback.
      
      Some ligand inputs (e.g. guanidinium / amidine groups, fused
      heteroaromatics with mixed Kekulé/aromatic perception, formal-charge
      nitrogens) cannot be kekulized or pass RDKit's strict valence model
      on first try. We retry with sanitization that skips KEKULIZE,
      SETAROMATICITY, and PROPERTIES — preserving the incoming bond
      orders / aromaticity flags rather than dropping the mol on the floor.
      
      When the molecule carries explicit aromatic flags from the source
      (see :func:`source_carried_kekule`), we skip ``SETAROMATICITY`` /
      ``KEKULIZE`` from the start so RDKit's re-perception cannot
      overwrite the source-supplied flags / Kekulé bond orders.
      

.. py:function:: kekulize_tolerant(mol: rdkit.Chem.Mol) -> None

   .. rubric:: Docstring

   .. code-block:: text

      Force Kekulé bond orders + clear aromatic flags, tolerant of failure.
      
      Rosetta's atom-type classifier and the reference ``.tmol`` files use
      Kekulé conventions (sp2 ``CD/CD1/CDp`` rather than aromatic
      ``CR/CRp``; explicit ``DOUBLE``/``SINGLE`` ring bonds rather than
      ``AROMATIC``). Aromaticity perception in RDKit's standard sanitize
      flips them back, so we kekulize after every sanitize step.
      

.. py:function:: should_kekulize_for_typing(mol: rdkit.Chem.Mol) -> bool

   .. rubric:: Docstring

   .. code-block:: text

      ``True`` when this mol's rings should use Kekulé typing.
      
      Rosetta's mol2genparams takes atom types directly from the source
      mol2's atom-type column. mol2 files written with sp2 (``C.2``) want
      Kekulé / ``CD`` typing; SMILES sources and mol2 files with ``C.ar``
      want aromatic / ``CR``. We can't recover the column itself, but we
      can read a flag set when the source AtomArray carried explicit
      Kekulé bond orders (see ``rdkit_mol.source_carried_kekule``).
      

.. py:function:: assign_tmol_atom_types(mol: rdkit.Chem.Mol, return_state: bool = False) -> list[AtomTypeAssignment] | tuple[list[AtomTypeAssignment], RosettaTypingState]

   .. rubric:: Docstring

   .. code-block:: text

      Assign Rosetta generic_potential atom types to each atom in a Mol.
      
      Follows the exact classification logic from Rosetta's AtomTypeClassifier
      (mol2genparams), including the polar-carbon modifier and ring-nitrogen
      corrections. Atom names follow Rosetta's rename_atoms convention:
      heavy atoms as <Element><count>, hydrogens as H<bonded_element><count>.
      

