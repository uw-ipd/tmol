tmol.ligand.preparation
=======================

.. py:module:: tmol.ligand.preparation

.. rubric:: Module docstring

.. code-block:: text

   Ligand preparation implementation for tmol.
   
   This module contains the concrete preparation pipeline implementation.
   `tmol.ligand.__init__` re-exports the public API from here.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.preparation.logger


Exceptions
----------

.. autoapisummary::

   tmol.ligand.preparation.LigandPreparationError


Functions
---------

.. autoapisummary::

   tmol.ligand.preparation.prepare_single_ligand
   tmol.ligand.preparation.prepare_ligands
   tmol.ligand.preparation.prepare_ligand_from_cif
   tmol.ligand.preparation.prepare_ligand_from_smiles
   tmol.ligand.preparation.prepare_ligand_from_mol2


Module Contents
---------------

.. py:data:: logger

.. py:exception:: LigandPreparationError

   Bases: :py:obj:`RuntimeError`


   .. rubric:: Docstring

   .. code-block:: text

      A detected ligand could not be prepared, registered, or retained.
      
      Raised by :func:`prepare_ligands` (and the ``prepare_ligands=True`` IO
      paths) when ``strict_ligands=True`` and a non-standard residue is skipped
      or fails preparation, instead of silently dropping it. Pass
      ``strict_ligands=False`` to downgrade these failures to warnings.
      

.. py:function:: prepare_single_ligand(ligand_info: tmol.ligand.detect.NonStandardResidueInfo, sample_proton_chi: bool = True, name_source: Optional[tmol.ligand.detect.NonStandardResidueInfo] = None) -> tmol.ligand.registry.LigandPreparation

   .. rubric:: Docstring

   .. code-block:: text

      Build a :class:`LigandPreparation` from a SMILES-derived ligand.
      
      This is the final, naming-and-typing step of the unified pipeline. Its input
      must already be fully resolved chemistry: explicit hydrogens at the desired
      protonation state and authoritative per-atom partial charges (the OpenBabel
      MMFF94 charges produced by the SMILES -> mol2 step). Protonation and charge
      generation are *not* done here -- they happen upstream in
      :func:`tmol.ligand.detect.nonstandard_residue_info_from_smiles_via_mol2`.
      
      Charges are mapped onto atoms by stable RDKit index (source atom order),
      so they are independent of the atom renaming below and never recomputed.
      
      Returns a :class:`LigandPreparation` -- the same struct
      :func:`tmol.ligand.params_file.load_params_file` produces for each residue
      defined in a ``.tmol`` file, so the AtomArray-driven path and the params-file
      path converge on a single abstraction that
      :func:`inject_ligand_preparations` consumes.
      
      :param ligand_info: A SMILES-derived ligand (``skip_protonation=True`` with
                          authoritative ``partial_charges``). Raw CIF/atom-array ligands must
                          be routed through :func:`prepare_ligands` / :func:`prepare_ligand_from_cif`.
      :param sample_proton_chi: Whether to emit proton-chi samples.
      :param name_source: Optional ligand whose atom names the prepared residue should
                          adopt (mapped to the prepared heavy atoms via the atom-order map). On
                          the unified CIF path this is the original CIF ligand. Defaults to
                          ``ligand_info``.
      
      :raises ValueError: If ``ligand_info`` lacks explicit hydrogens / authoritative
          charges (there is no charge-generation fallback).
      

.. py:function:: prepare_ligands(atom_array: biotite.structure.AtomArray, param_db: Optional[tmol.database.ParameterDatabase] = None, ph: float = 7.4, strict_atom_types: bool = False, params_files: list[str] | None = None, params_output: str | None = None, sample_proton_chi: bool = True, strict_ligands: bool = True, return_fragment_definitions: bool = False) -> tuple

   .. rubric:: Docstring

   .. code-block:: text

      Detect, prepare, and register all non-standard residues.
      
      Scans the input AtomArray for residues not in the ParameterDatabase,
      runs each through the unified SMILES→OpenBabel mol2→typing→residue-build
      pipeline, and returns a **new** ParameterDatabase with the ligand data
      injected.
      
      :param atom_array: A biotite AtomArray from a CIF or PDB file.
      :param param_db: The base ParameterDatabase (not modified). If None, the
                       default database is used.
      :param ph: Target pH for ligand protonation (Dimorphite-DL on derived SMILES).
      :param strict_atom_types: If True, fail when unknown atom-type element
                                mappings are encountered during registration.
      :param params_files: Optional list of tmol YAML params file paths to
                           inject before detection. Residues defined in these files
                           skip the RDKit/OB preparation pipeline.
      :param params_output: Optional path to write all prepared ligand data
                            to a tmol YAML params file for later reuse.
      :param sample_proton_chi: Whether to emit PROTON_CHI samples in the
                                built residue type.
      :param strict_ligands: If True (default), raise :class:`LigandPreparationError`
                             when a detected non-standard residue is skipped (metal-containing or
                             covalently linked) or fails preparation, instead of silently
                             dropping it. If False, such residues are logged as warnings and
                             skipped, leaving them to be filtered out during pose construction.
      :param return_fragment_definitions: Internal/context-building option. If True,
                                          include definitions derived from ``tmol_fragment_id`` annotations
                                          as the third return value.
      
      :returns: A (ParameterDatabase, CanonicalOrdering) tuple. When
                ``return_fragment_definitions`` is true, a third element containing
                the structure-independent ligand fragment definitions is returned. The returned
                ParameterDatabase is a new instance with all detected ligands
                injected; the input ``param_db`` is not modified.
      
      :raises LigandPreparationError: If ``strict_ligands`` and any detected ligand
          cannot be prepared and registered.
      

.. py:function:: prepare_ligand_from_cif(cif_path: str, *, param_db: Optional[tmol.database.ParameterDatabase] = None, ph: float = 7.4, strict_atom_types: bool = False, res_name: str | None = None, sample_proton_chi: bool = True) -> tuple[tmol.database.ParameterDatabase, tmol.io.canonical_ordering.CanonicalOrdering]

   .. rubric:: Docstring

   .. code-block:: text

      Prepare a single ligand from a CIF file and inject it into a database.
      
      Runs the same full pipeline as :func:`prepare_ligand_from_smiles`; the only
      CIF-specific step is the front end. A SMILES is derived from the CIF ligand's
      explicit bond table (never geometry perception, never a CCD lookup) and run
      through the SMILES -> mol2 -> params path (protonation, 3D conformer, MMFF94
      charges). The prepared residue's heavy-atom names are then mapped back to the
      CIF atom names via the atom-order map carried through the round-trip.
      
      :param cif_path: Path to the ligand CIF file.
      :param param_db: Base database (not modified); defaults to the tmol default.
      :param ph: Target pH for protonation.
      :param strict_atom_types: Fail on unknown atom-type element mappings.
      :param res_name: Optional residue name override.
      :param sample_proton_chi: Whether to emit proton-chi samples.
      
      :returns: A ``(ParameterDatabase, CanonicalOrdering)`` with the ligand injected.
      

.. py:function:: prepare_ligand_from_smiles(smiles: str, *, param_db: Optional[tmol.database.ParameterDatabase] = None, ph: float = 7.4, strict_atom_types: bool = False, res_name: str | None = None, protonate: bool = True, sample_proton_chi: bool = True, seed: int | None = None) -> tuple[tmol.database.ParameterDatabase, tmol.io.canonical_ordering.CanonicalOrdering]

   .. rubric:: Docstring

   .. code-block:: text

      Prepare a single ligand from a SMILES string and inject it into a database.
      
      Follows the canonical ligand-prep protocol: Dimorphite-DL pKa-protonates
      the SMILES at ``ph``, OpenBabel generates a 3D mol2 with MMFF94 partial
      charges, and that mol2 is read verbatim (atom names, coordinates, charges,
      and bond orders preserved). The MMFF94 charges flow through untouched —
      there is no biotite atom-array round-trip or MMFF recompute. This path
      requires the optional ``openbabel`` package.
      
      :param protonate: When ``True`` (default) Dimorphite protonates ``smiles``
                        first; set ``False`` to pin an already-protonated SMILES verbatim.
      :param seed: Fixed RNG seed for reproducible 3D coordinates; ``None`` is random.
      

.. py:function:: prepare_ligand_from_mol2(mol2_path: str, *, param_db: Optional[tmol.database.ParameterDatabase] = None, strict_atom_types: bool = False, res_name: str | None = None, sample_proton_chi: bool = True) -> tuple[tmol.database.ParameterDatabase, tmol.io.canonical_ordering.CanonicalOrdering]

   .. rubric:: Docstring

   .. code-block:: text

      Prepare a single ligand from a Tripos mol2 file and inject it.
      
      Reads atom names, coordinates, bond orders, and MMFF94 partial charges
      verbatim from the mol2 (no SMILES or OpenBabel 3D generation step).
      
      :param mol2_path: Path to the ligand mol2 file.
      :param param_db: Base database (not modified); defaults to the tmol default.
      :param strict_atom_types: Fail on unknown atom-type element mappings.
      :param res_name: Optional residue name override.
      :param sample_proton_chi: Whether to emit proton-chi samples.
      
      :returns: A ``(ParameterDatabase, CanonicalOrdering)`` with the ligand injected.
      

