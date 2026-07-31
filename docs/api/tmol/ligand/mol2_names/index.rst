tmol.ligand.mol2_names
======================

.. py:module:: tmol.ligand.mol2_names

.. rubric:: Module docstring

.. code-block:: text

   Rosetta-style disambiguation of duplicate Tripos atom names in MOL2 files.
   
   Rosetta ``mol2genparams`` renames repeated atom names in the same residue
   (e.g. a second ``C2'`` becomes ``C2'2``). PLI fixtures such as ``fgfr1`` rely
   on this when the same label appears on distinct atoms in one mol2 block.
   


Functions
---------

.. autoapisummary::

   tmol.ligand.mol2_names.disambiguate_mol2_atom_name
   tmol.ligand.mol2_names.apply_disambiguated_mol2_names


Module Contents
---------------

.. py:function:: disambiguate_mol2_atom_name(name: str, occurrence: int) -> str

   .. rubric:: Docstring

   .. code-block:: text

      Return the Rosetta/mol2gen name for the ``occurrence``-th use of ``name``.
      
      The first occurrence keeps the original name; the second becomes
      ``{name}2``, the third ``{name}3``, etc.
      

.. py:function:: apply_disambiguated_mol2_names(mol: rdkit.Chem.Mol) -> list[str]

   .. rubric:: Docstring

   .. code-block:: text

      Assign unique ``_TriposAtomName`` values on ``mol`` (mol2gen convention).
      
      Names are assigned in RDKit atom-index order, matching the order of the
      ``@<TRIPOS>ATOM`` block in the source file.
      
      :returns: The disambiguated name for each atom index.
      

