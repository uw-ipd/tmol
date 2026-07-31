tmol.ligand.rdkit_mol
=====================

.. py:module:: tmol.ligand.rdkit_mol

.. rubric:: Module docstring

.. code-block:: text

   RDKit molecule construction for ligands.
   
   Builds an RDKit ``Mol`` from a ligand ``AtomArray`` while preserving the
   source's explicit bond orders and aromatic/subtype annotations. Protonation
   and partial-charge generation are handled upstream by the SMILES -> OpenBabel
   mol2 step (:func:`tmol.ligand.detect.nonstandard_residue_info_from_smiles_via_mol2`),
   so this module does not protonate or recompute chemistry.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.rdkit_mol.logger


Functions
---------

.. autoapisummary::

   tmol.ligand.rdkit_mol.normalize_non_ring_aromatic_bonds
   tmol.ligand.rdkit_mol.normalize_cumulated_azide
   tmol.ligand.rdkit_mol.source_subtype
   tmol.ligand.rdkit_mol.source_carried_kekule
   tmol.ligand.rdkit_mol.source_has_aromatic_annotations
   tmol.ligand.rdkit_mol.rdkit_mol_from_ligand_atom_array
   tmol.ligand.rdkit_mol.ligand_atom_array_to_rdkit_mol


Module Contents
---------------

.. py:data:: logger

.. py:function:: normalize_non_ring_aromatic_bonds(mol: rdkit.Chem.Mol) -> None

   .. rubric:: Docstring

   .. code-block:: text

      Normalize non-ring aromatic placeholders before RDKit sanitize.
      

.. py:function:: normalize_cumulated_azide(mol: rdkit.Chem.Mol) -> rdkit.Chem.Mol

   .. rubric:: Docstring

   .. code-block:: text

      Strip a spurious H from a charge-separated azide/diazo terminus.
      
      Convert N=N=N-H (which RDKit does not understand) to =[N+]=[N-]
      Do nothing if this group is not found.
      

.. py:function:: source_subtype(atom: rdkit.Chem.Atom) -> str

   .. rubric:: Docstring

   .. code-block:: text

      Return the source mol2 atom-type subtype tag (e.g. ``ar``, ``2``,
      ``cat``, ``pl3``, ``3``) when known, else ``""``.
      

.. py:function:: source_carried_kekule(mol: rdkit.Chem.Mol) -> bool

   .. rubric:: Docstring

   .. code-block:: text

      True iff the source molecule was constructed with Kekulé bond orders.
      
      Set by :func:`_restore_kekule_bonds` when the input AtomArray carried
      explicit ``SINGLE`` / ``DOUBLE`` (or biotite's ``AROMATIC_SINGLE`` /
      ``AROMATIC_DOUBLE``) ring bonds — typical for mol2 files written
      with ``C.2`` (sp2). SMILES inputs come through with only
      ``AROMATIC`` bonds and leave this flag unset.
      

.. py:function:: source_has_aromatic_annotations(mol: rdkit.Chem.Mol) -> bool

   .. rubric:: Docstring

   .. code-block:: text

      True iff aromatic atom flags were provided by the source input.
      

.. py:function:: rdkit_mol_from_ligand_atom_array(atom_array: biotite.structure.AtomArray, *, res_name: str = 'ligand', keep_hydrogens: bool = False, repair_chemistry: bool = False) -> rdkit.Chem.Mol

   .. rubric:: Docstring

   .. code-block:: text

      Build an RDKit Mol from a ligand AtomArray's explicit bond table.
      
      The single AtomArray -> RDKit builder for the ligand pipeline: it preserves
      the source's explicit bond orders (restoring Kekulé forms biotite's
      ``to_mol`` collapses) and aromatic/subtype annotations. Bond perception from
      geometry is intentionally unsupported — the input must carry chemistry-level
      bond orders.
      
      :param atom_array: The ligand sub-array (heavy + optional hydrogen atoms).
      :param res_name: Residue code, used only for log/error messages.
      :param keep_hydrogens: When True, retain explicit hydrogens from the input
                             (used for ``skip_protonation`` — preserve mol2/CIF protonation).
      :param repair_chemistry: When True, apply last-resort chemistry normalizations
                               that rewrite source bond orders to make an otherwise-unrepresentable
                               input build.
      

.. py:function:: ligand_atom_array_to_rdkit_mol(ligand_info: tmol.ligand.detect.NonStandardResidueInfo, *, keep_hydrogens: bool = False) -> rdkit.Chem.Mol

   .. rubric:: Docstring

   .. code-block:: text

      Build an RDKit Mol from a detected ligand's AtomArray.
      
      Thin wrapper over :func:`rdkit_mol_from_ligand_atom_array`.
      

