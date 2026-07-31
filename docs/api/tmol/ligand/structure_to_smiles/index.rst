tmol.ligand.structure_to_smiles
===============================

.. py:module:: tmol.ligand.structure_to_smiles

.. rubric:: Module docstring

.. code-block:: text

   Derive a ligand SMILES string from a biotite ``AtomArray``.
   
   This is the entry to tmol's unified ligand path: a CIF/atom-array ligand is
   converted to a SMILES string here, then handed to the existing
   SMILES -> params pipeline (:func:`nonstandard_residue_info_from_smiles_via_mol2`).
   
   The SMILES always reflects the *input atoms as given* -- there is no residue-code
   / CCD-template lookup (that risks substituting an unrelated molecule when a CIF
   uses a generic residue code such as ``LG1``).
   
   Bond orders must come from the input: the ``AtomArray`` is required to carry a
   bond table (e.g. a CIF with a ``_chem_comp_bond`` block, or a mol2 BOND
   section). We deliberately do *not* perceive bonds from 3D geometry -- a
   bonds-absent input (such as a plain PDB ligand) is a hard error, because guessed
   bond orders would silently corrupt the generated params database.
   
   The SMILES is built with the shared ligand builder
   :func:`tmol.ligand.rdkit_mol.rdkit_mol_from_ligand_atom_array` -- the same
   AtomArray -> RDKit path the params pipeline uses -- so the derived SMILES and
   the prepared molecule always agree on chemistry.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.structure_to_smiles.logger


Functions
---------

.. autoapisummary::

   tmol.ligand.structure_to_smiles.apply_geometry_bond_corrections
   tmol.ligand.structure_to_smiles.ligand_smiles_from_atom_array


Module Contents
---------------

.. py:data:: logger

.. py:function:: apply_geometry_bond_corrections(mol: rdkit.Chem.Mol) -> rdkit.Chem.Mol

   .. rubric:: Docstring

   .. code-block:: text

      Repair input bond orders that disagree with the 3D geometry.
      
      Runs each geometry-based correction rule (carboxylate only, for now) and
      re-sanitizes. Returns the input unchanged when there is no conformer or no
      correction applies. More rules (nitro, phosphate, sulfonate, ...) can be
      added as separate ``_infer_*`` functions and dispatched here.
      

.. py:function:: ligand_smiles_from_atom_array(atom_array: biotite.structure.AtomArray, *, res_name: str | None = None, with_atom_map: bool = False) -> str

   .. rubric:: Docstring

   .. code-block:: text

      Derive a canonical SMILES for a ligand AtomArray from its bond table.
      
      The SMILES is derived purely from the input atoms and their explicit bonds
      (never a residue-code / CCD-template lookup, never geometry-based bond
      perception). Geometry-based bond-*order* corrections are still applied for
      motifs the input encodes inconsistently (carboxylates).
      
      :param atom_array: The ligand sub-array (heavy + optional hydrogen atoms).
      :param res_name: Residue code, used only for log/error messages.
      :param with_atom_map: Tag heavy atoms with source-index map numbers for CIF
                            atom naming downstream.
      
      :returns: A canonical SMILES string.
      
      :raises ValueError: If the AtomArray carries no bond table (bond orders must be
          supplied by the input; a bonds-absent ligand such as a plain PDB
          cannot be prepared without guessing chemistry), or if no SMILES
          could be derived from the bonds present.
      

