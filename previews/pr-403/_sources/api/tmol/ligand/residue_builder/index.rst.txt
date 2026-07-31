tmol.ligand.residue_builder
===========================

.. py:module:: tmol.ligand.residue_builder

.. rubric:: Module docstring

.. code-block:: text

   Build tmol RawResidueType definitions from RDKit molecules.
   
   Converts a Chem.Mol with assigned atom types into a complete RawResidueType
   suitable for registration in tmol's ChemicalDatabase. Handles atom tree
   construction, internal coordinate computation, rotatable bond detection,
   and non-polymer property assignment.
   
   The internal coordinate and atom tree algorithms are ported from
   Rosetta's mol2genparams.py and molfile_to_params.py.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.residue_builder.logger


Functions
---------

.. autoapisummary::

   tmol.ligand.residue_builder.build_residue_type


Module Contents
---------------

.. py:data:: logger

.. py:function:: build_residue_type(mol: rdkit.Chem.Mol, res_name: str, atom_types: list[tmol.ligand.atom_typing.AtomTypeAssignment], atom_aliases: tuple = (), *, typing_state=None, sample_proton_chi: bool = True, original_single_bonds: frozenset[frozenset[str]] | None = None) -> tmol.database.chemical.RawResidueType

   .. rubric:: Docstring

   .. code-block:: text

      Build a complete RawResidueType from a Chem.Mol.
      
      Constructs atoms, bonds, internal coordinates, and non-polymer
      properties suitable for registration in tmol's ChemicalDatabase.
      
      Atoms with unknown elements (atomic number 0, e.g. metals that
      lost identity during SMILES roundtrip) are silently dropped.
      
      :param mol: An RDKit Mol with 3D coordinates and bonds.
      :param res_name: Three-letter residue name (e.g. "LG1", "ATP").
      :param atom_types: Atom type assignments from assign_tmol_atom_types().
      :param atom_aliases: Optional tuple of AtomAlias for CIF name mapping.
      
      :returns: A fully populated RawResidueType.
      

