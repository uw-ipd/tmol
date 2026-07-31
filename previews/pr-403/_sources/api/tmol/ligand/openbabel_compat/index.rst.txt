tmol.ligand.openbabel_compat
============================

.. py:module:: tmol.ligand.openbabel_compat

.. rubric:: Module docstring

.. code-block:: text

   OpenBabel fallback helpers for RDKit-fragile input formats.
   
   RDKit's mol2 / SMILES / PDB parsers reject many real-world ligand files
   that OpenBabel handles fine. The helpers in this module call OpenBabel
   to read the input, then re-emit it as an SDF mol-block string that RDKit
   can ingest. The result is an ``rdkit.Chem.Mol`` indistinguishable from
   one produced by a successful RDKit parse, so downstream code (atom
   typing, residue building, scoring) needs no further changes.
   
   OpenBabel is a *soft* dependency. Import happens inside each helper, so
   the rest of ``tmol.ligand`` loads cleanly on systems without
   ``openbabel-wheel``. The helpers raise a single descriptive error if a
   caller invokes them without OB installed.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.openbabel_compat.logger


Exceptions
----------

.. autoapisummary::

   tmol.ligand.openbabel_compat.OpenBabelUnavailableError


Functions
---------

.. autoapisummary::

   tmol.ligand.openbabel_compat.strip_nontetrahedral_stereo
   tmol.ligand.openbabel_compat.normalize_azide
   tmol.ligand.openbabel_compat.source_atom_order_from_mapped_smiles
   tmol.ligand.openbabel_compat.obabel_read_mol2
   tmol.ligand.openbabel_compat.obabel_read_mol2_block
   tmol.ligand.openbabel_compat.obabel_smiles_to_mol2_block
   tmol.ligand.openbabel_compat.obabel_smiles_to_mol2


Module Contents
---------------

.. py:data:: logger

.. py:exception:: OpenBabelUnavailableError

   Bases: :py:obj:`RuntimeError`


   .. rubric:: Docstring

   .. code-block:: text

      Raised when an OB-fallback helper is called but ``openbabel`` is missing.
      

.. py:function:: strip_nontetrahedral_stereo(smiles: str) -> str

   .. rubric:: Docstring

   .. code-block:: text

      Drop stereo descriptors OpenBabel cannot parse, keeping the rest.
      
      Returns ``smiles`` unchanged if it has no such markers or cannot be parsed.
      

.. py:function:: normalize_azide(smiles: str) -> str

   .. rubric:: Docstring

   .. code-block:: text

      Rewrite the neutral cumulated azide ``N=[N]=N`` (CACTVS notation) to the
      charged form ``N=[N+]=[N-]`` that RDKit can sanitize.
      

.. py:function:: source_atom_order_from_mapped_smiles(smiles: str) -> Optional[tuple[int, Ellipsis]]

   .. rubric:: Docstring

   .. code-block:: text

      Map numbers (source indices) in SMILES/mol2 atom order, or None if any
      atom is unmapped. Normalized identically to the mol2 path so order matches.
      

.. py:function:: obabel_read_mol2(path: str | pathlib.Path) -> Optional[rdkit.Chem.Mol]

   .. rubric:: Docstring

   .. code-block:: text

      Read a TRIPOS mol2 file via OpenBabel and return an RDKit ``Chem.Mol``.
      
      Use as a fallback when ``Chem.MolFromMol2File`` returns ``None`` on
      valid mol2 files that RDKit's parser rejects.
      
      Returns ``None`` if OB could not parse the file. Raises
      :class:`OpenBabelUnavailableError` if OpenBabel is not installed.
      

.. py:function:: obabel_read_mol2_block(mol2_block: str) -> Optional[rdkit.Chem.Mol]

   .. rubric:: Docstring

   .. code-block:: text

      Read a TRIPOS mol2 *string* via OpenBabel and return an RDKit ``Chem.Mol``.
      
      In-memory analogue of :func:`obabel_read_mol2` — use as a fallback when
      ``Chem.MolFromMol2Block`` returns ``None``. Returns ``None`` if OB could not
      parse the block. Raises :class:`OpenBabelUnavailableError` if OB is missing.
      

.. py:function:: obabel_smiles_to_mol2_block(smiles: str, *, forcefield: str = 'mmff94', minimize_steps: int = 50, seed: Optional[int] = None) -> str

   .. rubric:: Docstring

   .. code-block:: text

      Return a 3D MMFF94 mol2 as an in-memory TRIPOS string (no disk I/O).
      
      Preferred over :func:`obabel_smiles_to_mol2` for high-throughput batches
      (e.g. millions of SMILES): the mol2 is handed downstream as a string rather
      than written to and re-read from a temp file. See
      :func:`_build_charged_3d_mol2_mol` for the protocol and raised errors.
      

.. py:function:: obabel_smiles_to_mol2(smiles: str, out_path: str | pathlib.Path, *, forcefield: str = 'mmff94', minimize_steps: int = 50, seed: Optional[int] = None) -> pathlib.Path

   .. rubric:: Docstring

   .. code-block:: text

      Generate a 3D MMFF94 mol2 from a SMILES and write it to ``out_path``.
      
      File-writing wrapper around :func:`obabel_smiles_to_mol2_block`; prefer the
      block form when no on-disk mol2 is required. See
      :func:`_build_charged_3d_mol2_mol` for the protocol and raised errors.
      
      :returns: The ``out_path`` written.
      

