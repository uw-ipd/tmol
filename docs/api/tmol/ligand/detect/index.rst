tmol.ligand.detect
==================

.. py:module:: tmol.ligand.detect

.. rubric:: Module docstring

.. code-block:: text

   Detection of non-standard residues in biotite AtomArrays.
   
   Identifies residues that are not represented in tmol's ChemicalDatabase
   and classifies them using Biotite's built-in Chemical Component Dictionary
   (CCD) as either true ligands (non-polymer) or modified amino acids /
   nucleotides (polymer-linked).
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.detect.logger
   tmol.ligand.detect.SKIP_RESIDUES


Classes
-------

.. autoapisummary::

   tmol.ligand.detect.NonStandardResidueInfo


Functions
---------

.. autoapisummary::

   tmol.ligand.detect.get_chem_comp_type
   tmol.ligand.detect.nonstandard_residue_info_from_mol2
   tmol.ligand.detect.nonstandard_residue_info_from_mol2_block
   tmol.ligand.detect.nonstandard_residue_info_from_smiles_via_mol2
   tmol.ligand.detect.detect_nonstandard_residues


Module Contents
---------------

.. py:data:: logger

.. py:data:: SKIP_RESIDUES

.. py:class:: NonStandardResidueInfo

   .. rubric:: Docstring

   .. code-block:: text

      Detected non-standard residue requiring preparation.
      
      Any residue not in tmol's standard database is represented here,
      regardless of whether it is a true ligand, modified amino acid,
      or modified nucleotide.
      
      .. attribute:: res_name
      
         Three-letter residue code (e.g. "ATP", "NAG").
      
      .. attribute:: ccd_type
      
         CCD chemical component type string, or "UNKNOWN" if the
         residue is not in the CCD.  Informational only.
      
      .. attribute:: atom_names
      
         Atom names for one representative instance.
      
      .. attribute:: elements
      
         Element symbols for each atom.
      
      .. attribute:: coords
      
         Cartesian coordinates of shape (n_atoms, 3).
      
      .. attribute:: atom_array
      
         The sub-AtomArray (with bonds if available).
      
      .. attribute:: partial_charges
      
         Authoritative ``{atom_name: charge}`` map (OpenBabel
         MMFF94 charges). Set only on the mol2 / SMILES-via-mol2 reader path,
         where ``prepare_single_ligand`` consumes them directly. ``None`` for
         raw CIF/atom-array detections (the unified path re-derives charges
         from the SMILES).
      
      .. attribute:: skip_protonation
      
         If True, Dimorphite-DL protonation is skipped and
         explicit hydrogens from the input (mol2) are preserved. Paired with
         ``partial_charges`` on the mol2 / SMILES-via-mol2 path.
      
      .. attribute:: original_single_bonds
      
         Optional set of ``frozenset({name_a, name_b})``
         pairs that the source mol2 records as literal single (order ``'1'``)
         bonds, keyed by disambiguated atom name. ``build_chi_topology`` uses
         these to honor the mol2 bond order (Rosetta-faithful) instead of
         RDKit's post-kekulization order. Only set on the mol2 / SMILES-via-
         mol2 paths; ``None`` elsewhere.
      

   .. py:attribute:: res_name
      :type:  str


   .. py:attribute:: ccd_type
      :type:  str


   .. py:attribute:: atom_names
      :type:  tuple[str, Ellipsis]


   .. py:attribute:: elements
      :type:  tuple[str, Ellipsis]


   .. py:attribute:: coords
      :type:  numpy.ndarray


   .. py:attribute:: atom_array
      :type:  biotite.structure.AtomArray


   .. py:attribute:: covalently_linked
      :type:  bool
      :value: False



   .. py:attribute:: partial_charges
      :type:  Optional[dict[str, float]]
      :value: None



   .. py:attribute:: skip_protonation
      :type:  bool
      :value: False



   .. py:attribute:: original_single_bonds
      :type:  Optional[frozenset[frozenset[str]]]
      :value: None



   .. py:attribute:: source_atom_order
      :type:  Optional[tuple[int, Ellipsis]]
      :value: None



.. py:function:: get_chem_comp_type(res_name: str) -> Optional[str]

   .. rubric:: Docstring

   .. code-block:: text

      Look up the CCD chemical component type for a residue name.
      
      :param res_name: Three-letter residue code.
      
      :returns: The CCD type string (e.g. "NON-POLYMER", "L-PEPTIDE LINKING"),
                or None if the code is not found in the CCD.
      

.. py:function:: nonstandard_residue_info_from_mol2(mol2_path: str | pathlib.Path, res_name: str | None = None) -> NonStandardResidueInfo

   .. rubric:: Docstring

   .. code-block:: text

      Construct ``NonStandardResidueInfo`` from a ligand Mol2 file.
      
      Low-level reader retained for the DUD-80 SMILES->params parity harness
      (it reads both the OpenBabel-generated and Rosetta ground-truth mol2
      files). Preserves Tripos aromatic flags, atom-type subtypes, and per-atom
      partial charges, avoiding lossy rdkit<->biotite round-trips.
      

.. py:function:: nonstandard_residue_info_from_mol2_block(mol2_block: str, res_name: str | None = None) -> NonStandardResidueInfo

   .. rubric:: Docstring

   .. code-block:: text

      Construct ``NonStandardResidueInfo`` from an in-memory mol2 *string*.
      
      In-memory analogue of :func:`nonstandard_residue_info_from_mol2` — parses a
      TRIPOS mol2 block directly, with no temp-file write/read. Preferred for
      high-throughput SMILES batches (see
      :func:`nonstandard_residue_info_from_smiles_via_mol2`).
      

.. py:function:: nonstandard_residue_info_from_smiles_via_mol2(smiles: str, res_name: str | None = None, *, ph: float = 7.4, protonate: bool = True, seed: int | None = None) -> NonStandardResidueInfo

   .. rubric:: Docstring

   .. code-block:: text

      Construct ``NonStandardResidueInfo`` from a SMILES via the mol2 route.
      
      Implements the canonical ligand-prep protocol end to end:
      
      0. normalize bare radical oxygens (``[O]`` -> ``[O-]``) so source
         carboxylate/sulfonate notation has a well-defined charge,
      1. optionally pKa-protonate the SMILES with Dimorphite-DL (``protonate``),
      2. generate a 3D mol2 with MMFF94 partial charges via OpenBabel (kept
         in memory as a string — no temp file), then
      3. read that mol2 with :func:`nonstandard_residue_info_from_mol2_block`.
      
      This never builds a biotite atom-array from an RDKit embedding and never
      recomputes MMFF on a reconstructed graph — the OpenBabel MMFF94 charges
      flow through untouched (``skip_protonation`` / authoritative charges are
      set by the mol2 reader), so fused-ring aromatics keep correct charges.
      
      :param smiles: Ligand SMILES string.
      :param res_name: Three-letter residue name (default inferred / ``"LG1"``).
      :param ph: Target pH for the Dimorphite protonation step.
      :param protonate: When ``True`` (default) run Dimorphite on ``smiles`` first;
                        set ``False`` to pin an already-protonated SMILES verbatim.
      :param seed: Fixed RNG seed for reproducible 3D coordinates; ``None`` is random.
      
      :raises OpenBabelUnavailableError: If the ``openbabel`` package is missing
          (this path requires it for the SMILES -> mol2 conversion).
      :raises ValueError: If OpenBabel cannot build a charged mol2 for ``smiles``.
      

.. py:function:: detect_nonstandard_residues(atom_array: biotite.structure.AtomArray, canonical_ordering: tmol.io.canonical_ordering.CanonicalOrdering) -> list[NonStandardResidueInfo]

   .. rubric:: Docstring

   .. code-block:: text

      Detect residues in an AtomArray that are not in tmol's database.
      
      Any residue whose 3-letter code is not in the canonical ordering
      is returned for preparation, regardless of whether it is a ligand,
      modified amino acid, or modified nucleotide.
      
      :param atom_array: Biotite AtomArray from a CIF or PDB file.
      :param canonical_ordering: The current tmol CanonicalOrdering, which
                                 defines known residue types.
      
      :returns: A list of NonStandardResidueInfo objects, one per unique unknown
                residue name.
      

