tmol.ligand.dimorphite_dl
=========================

.. py:module:: tmol.ligand.dimorphite_dl

.. rubric:: Module docstring

.. code-block:: text

   Dimorphite-DL: enumerate ionization states of drug-like small molecules.
   
   Identifies and enumerates the possible protonation sites of SMILES strings
   at a user-specified pH range using pre-calculated pKa distributions.
   
   Originally authored by Jacob D. Durrant (Dimorphite-DL 1.2.4). Vendored
   into tmol and cleaned up for lint compliance, type annotations, and
   Google-style docstrings. Protonation logic is unchanged.
   
   Reference:
       Ropp PJ, Kaminsky JC, Yablonski S, Durrant JD (2019) Dimorphite-DL: An
       open-source program for enumerating the ionization states of drug-like
       small molecules. J Cheminform 11:14. doi:10.1186/s13321-019-0336-9.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.dimorphite_dl.logger
   tmol.ligand.dimorphite_dl.msg


Classes
-------

.. autoapisummary::

   tmol.ligand.dimorphite_dl.MyParser
   tmol.ligand.dimorphite_dl.ArgParseFuncs
   tmol.ligand.dimorphite_dl.UtilFuncs
   tmol.ligand.dimorphite_dl.LoadSMIFile
   tmol.ligand.dimorphite_dl.Protonate
   tmol.ligand.dimorphite_dl.ProtSubstructFuncs
   tmol.ligand.dimorphite_dl.ProtectUnprotectFuncs
   tmol.ligand.dimorphite_dl.TestFuncs


Functions
---------

.. autoapisummary::

   tmol.ligand.dimorphite_dl.print_header
   tmol.ligand.dimorphite_dl.main
   tmol.ligand.dimorphite_dl.protonate_mol_variants


Module Contents
---------------

.. py:data:: logger

.. py:data:: msg
   :value: 'Dimorphite-DL requires RDKit. See https://www.rdkit.org/'


.. py:function:: print_header() -> None

   .. rubric:: Docstring

   .. code-block:: text

      Log citation and help information.
      

.. py:function:: main(params: dict[str, Any] | None = None) -> list[str] | None

   .. rubric:: Docstring

   .. code-block:: text

      Entry point when the script is called from the command line.
      
      :param params: Optional parameter dictionary. If absent, arguments are
                     parsed from the command line.
      
      :returns: A list of protonated SMILES strings when the ``return_as_list``
                parameter is True, otherwise None.
      

.. py:class:: MyParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)

   Bases: :py:obj:`argparse.ArgumentParser`


   .. rubric:: Docstring

   .. code-block:: text

      ArgumentParser subclass that prints help on error.
      

   .. py:method:: error(message: str) -> None

      .. rubric:: Docstring

      .. code-block:: text

         Print help and raise on parse error.
         
         :param message: The error message from argparse.
         
         :raises Exception: Always raised with the error message.
         


   .. py:method:: print_help(file: IO[str] | None = None) -> None

      .. rubric:: Docstring

      .. code-block:: text

         Print help text with usage examples.
         
         :param file: Output stream. Defaults to stdout.
         


.. py:class:: ArgParseFuncs

   .. rubric:: Docstring

   .. code-block:: text

      Namespace for command-line argument processing functions.
      

   .. py:method:: get_args() -> MyParser
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Build and return the argument parser.
         
         :returns: A configured argument parser instance.
         


   .. py:method:: clean_args(args: dict[str, Any]) -> dict[str, Any]
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Clean and normalise input parameters.
         
         Fills in defaults for missing keys, removes ``None`` values, and
         converts a bare ``smiles`` string into a file-like object.
         
         :param args: Mutable dictionary of arguments.
         
         :returns: The cleaned argument dictionary (same object, mutated).
         
         :raises Exception: If neither ``smiles`` nor ``smiles_file`` is provided.
         


.. py:class:: UtilFuncs

   .. rubric:: Docstring

   .. code-block:: text

      Namespace for molecular utility functions.
      

   .. py:method:: neutralize_mol(mol: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol | None
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Neutralise a molecule by iteratively applying SMARTS reactions.
         
         Removes inappropriate charges (e.g. O-, N+) and fixes azide
         representations. The user should not be allowed to specify atom
         valences in most cases.
         
         :param mol: The RDKit Mol object to neutralise.
         
         :returns: The neutralised Mol object, or None if sanitisation fails.
         


   .. py:method:: convert_smiles_str_to_mol(smiles_str: str | None) -> rdkit.Chem.rdchem.Mol | None
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Convert a SMILES string to an RDKit Mol object.
         
         Performs type checking, fixes common azide issues, and suppresses
         RDKit stderr output during conversion.
         
         :param smiles_str: The SMILES string to convert.
         
         :returns: An RDKit Mol object, or None on failure.
         


   .. py:method:: eprint(*args: Any, **kwargs: Any) -> None
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Log to stderr-equivalent channel.
         
         :param \*args: Positional arguments forwarded to logger.
         :param \*\*kwargs: Unused keyword arguments for compatibility.
         


.. py:class:: LoadSMIFile(filename: str | IO[str], args: dict[str, Any])

   .. rubric:: Docstring

   .. code-block:: text

      Generator that loads and pre-processes SMILES strings from a file.
      

   .. py:attribute:: args


   .. py:method:: next() -> dict[str, Any]

      .. rubric:: Docstring

      .. code-block:: text

         Read and process the next line from the SMILES file.
         
         Converts the raw SMILES to a canonical, neutralised form with
         hydrogens removed.
         
         :returns: A dict with ``"smiles"`` (canonical SMILES) and ``"data"``
                   (remaining tab-separated fields).
         
         :raises StopIteration: When the file is exhausted.
         


.. py:class:: Protonate(args: dict[str, Any])

   .. rubric:: Docstring

   .. code-block:: text

      Generator that yields protonated SMILES strings one at a time.
      

   .. py:attribute:: args


   .. py:attribute:: cur_prot_SMI
      :type:  list[str]
      :value: []



   .. py:attribute:: subs
      :value: []



   .. py:method:: next() -> str

      .. rubric:: Docstring

      .. code-block:: text

         Return the next protonated SMILES string.
         
         Handles multi-site protonation by expanding combinations and
         caching results in ``self.cur_prot_SMI``.
         
         :returns: A protonated SMILES string with optional label and tag.
         
         :raises StopIteration: When all input SMILES have been processed.
         


.. py:class:: ProtSubstructFuncs

   .. rubric:: Docstring

   .. code-block:: text

      Namespace for protonation-substructure matching and site modification.
      

   .. py:attribute:: args
      :type:  dict[str, Any]


   .. py:method:: load_substructre_smarts_file() -> list[str]
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Load the substructure SMARTS file, filtering out comments.
         
         :returns: Non-blank, non-comment lines from ``site_substructures.smarts``.
         


   .. py:method:: load_protonation_substructs_calc_state_for_ph(min_ph: float = 6.4, max_ph: float = 8.4, pka_std_range: float = 1) -> list[dict[str, Any]]
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Load protonation substructures and calculate states for a pH range.
         
         Reads the SMARTS definitions file and, for each protonation site,
         determines whether it should be protonated, deprotonated, or both
         at the given pH range.
         
         :param min_ph: Lower bound of the pH range.
         :param max_ph: Upper bound of the pH range.
         :param pka_std_range: Number of standard deviations from the mean pKa
                               to consider.
         
         :returns: A list of substructure dicts, each containing ``"name"``,
                   ``"smart"``, ``"mol"``, and ``"prot_states_for_pH"``.
         


   .. py:method:: define_protonation_state(mean: float, std: float, min_ph: float, max_ph: float) -> str
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Determine the protonation state for a site at a given pH range.
         
         :param mean: The mean pKa value.
         :param std: The standard deviation (precision).
         :param min_ph: Minimum pH of the range.
         :param max_ph: Maximum pH of the range.
         
         :returns: One of ``"PROTONATED"``, ``"DEPROTONATED"``, or ``"BOTH"``.
         


   .. py:method:: get_prot_sites_and_target_states(smi: str, subs: list[dict[str, Any]]) -> tuple[list[tuple[int, str, str]], rdkit.Chem.rdchem.Mol | None]
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Find protonation sites and their target states for a molecule.
         
         Matches the molecule against the substructure list. Sites higher
         in the list take priority and protect matched atoms from later
         matches.
         
         :param smi: A SMILES string.
         :param subs: Substructure definitions from
                      :func:`load_protonation_substructs_calc_state_for_ph`.
         
         :returns: A tuple of (sites, mol) where sites is a list of
                   ``(atom_index, target_state, site_name)`` tuples and mol is
                   the hydrogenated Mol object used for indexing.
         


   .. py:method:: get_prot_sites_and_target_states_from_mol(mol: rdkit.Chem.rdchem.Mol, subs: list[dict[str, Any]]) -> tuple[list[tuple[int, str, str]], rdkit.Chem.rdchem.Mol | None]
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Find protonation sites and target states for an RDKit Mol.
         
         :param mol: Input molecule.
         :param subs: Substructure definitions.
         
         :returns: A tuple of (sites, hydrogenated_mol). If processing fails,
                   returns ([], None).
         


   .. py:method:: protonate_site(mols: list[rdkit.Chem.rdchem.Mol], site: tuple[int, str, str]) -> list[rdkit.Chem.rdchem.Mol]
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Protonate or deprotonate a single site across a list of molecules.
         
         :param mols: Input molecule objects.
         :param site: A ``(atom_index, target_state, site_name)`` tuple.
         
         :returns: A list of molecule objects with the site adjusted.
         


   .. py:method:: set_protonation_charge(mols: list[rdkit.Chem.rdchem.Mol], idx: int, charges: list[int], prot_site_name: str) -> list[rdkit.Chem.rdchem.Mol]
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Set the formal charge at a protonation site for each molecule.
         
         Handles nitrogen, oxygen, and sulfur atoms with appropriate
         hydrogen counts based on the charge and bond order.
         
         :param mols: Input molecule objects.
         :param idx: Atom index of the protonation site.
         :param charges: List of charges to assign (one mol copy per charge).
         :param prot_site_name: Name of the protonation site definition.
         
         :returns: A list of molecule objects with charges assigned.
         


.. py:class:: ProtectUnprotectFuncs

   .. rubric:: Docstring

   .. code-block:: text

      Namespace for atom protection/unprotection during substructure matching.
      

   .. py:method:: unprotect_molecule(mol: rdkit.Chem.rdchem.Mol) -> None
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Mark all atoms in the molecule as unprotected.
         
         :param mol: The RDKit Mol object whose atoms to unprotect.
         


   .. py:method:: protect_molecule(mol: rdkit.Chem.rdchem.Mol, match: tuple[int, Ellipsis]) -> None
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Mark matched atoms as protected to prevent re-matching.
         
         :param mol: The RDKit Mol object.
         :param match: Tuple of atom indices to protect.
         


   .. py:method:: get_unprotected_matches(mol: rdkit.Chem.rdchem.Mol, substruct: rdkit.Chem.rdchem.Mol, site_indices: Optional[list[int]] = None) -> list[tuple[int, Ellipsis]]
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Find matches this rule is still allowed to ionize.
         
         :param mol: The molecule to search.
         :param substruct: The SMARTS substructure pattern.
         :param site_indices: Positions within the match that this rule would
                              actually ionize. When given, only those atoms are checked for
                              protection (see :func:`is_match_unprotected`).
         
         :returns: A list of matches (each a tuple of atom indices).
         


   .. py:method:: is_match_unprotected(mol: rdkit.Chem.rdchem.Mol, match: tuple[int, Ellipsis], site_indices: Optional[list[int]] = None) -> bool
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Check whether the atoms this rule would ionize are unprotected.
         
         (fd) ONLY protonation sites can block it;
              protected context atoms (carbons) carry no ionization decision.
         
         :param mol: The RDKit Mol object.
         :param match: Tuple of atom indices to check.
         :param site_indices: Positions within ``match`` that this rule ionizes.
                              When None, every matched atom is checked (legacy behavior).
         
         :returns: True if none of the checked atoms are protected.
         


.. py:class:: TestFuncs

   .. rubric:: Docstring

   .. code-block:: text

      Built-in self-tests for all 38 protonation groups.
      

   .. py:method:: test() -> None
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Run the full test suite for all ionisable groups.
         


   .. py:method:: test_check(args: dict[str, Any], expected_output: list[str], labels: list[str]) -> None
      :staticmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Verify protonation output against expected values.
         
         :param args: Arguments to pass to :class:`Protonate`.
         :param expected_output: Expected SMILES strings.
         :param labels: Expected state labels (``BOTH``, ``PROTONATED``,
                        ``DEPROTONATED``).
         
         :raises Exception: If the output doesn't match expectations.
         


.. py:function:: protonate_mol_variants(mol: rdkit.Chem.rdchem.Mol, min_ph: float = 6.4, max_ph: float = 8.4, pka_precision: float = 1.0, max_variants: int = 128, silent: bool = True) -> list[rdkit.Chem.rdchem.Mol]

   .. rubric:: Docstring

   .. code-block:: text

      Protonate an RDKit Mol directly, without a SMILES roundtrip.
      

