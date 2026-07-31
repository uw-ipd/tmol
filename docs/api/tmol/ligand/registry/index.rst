tmol.ligand.registry
====================

.. py:module:: tmol.ligand.registry

.. rubric:: Module docstring

.. code-block:: text

   Registration of dynamically created ligand residue types.
   
   Extends tmol's ParameterDatabase with new residue types and their
   scoring parameters built by the ligand preparation pipeline.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.registry.logger


Classes
-------

.. autoapisummary::

   tmol.ligand.registry.LigandPreparation


Functions
---------

.. autoapisummary::

   tmol.ligand.registry.collect_new_atom_types
   tmol.ligand.registry.inject_ligand_preparations
   tmol.ligand.registry.rebuild_canonical_ordering


Module Contents
---------------

.. py:data:: logger

.. py:function:: collect_new_atom_types(chem_db: tmol.chemical.patched_chemdb.PatchedChemicalDatabase, residue_type: tmol.database.chemical.RawResidueType, atom_type_elements: Optional[dict[str, str]] = None, *, strict_atom_types: bool = False) -> list[tmol.database.chemical.AtomType]

   .. rubric:: Docstring

   .. code-block:: text

      Identify atom types used by the residue that aren't in the database.
      
      Sets hbond properties (is_donor, is_acceptor, acceptor_hybridization)
      from the HBOND_PROPERTIES lookup in atom_typing.py.
      

.. py:class:: LigandPreparation

   .. rubric:: Docstring

   .. code-block:: text

      The unified abstraction both ligand-pipeline paths converge on.
      
      A ``LigandPreparation`` is everything tmol needs to inject one ligand
      into a ``ParameterDatabase``: the residue type definition, partial
      charges, cartbonded parameters, and (optionally) the element mapping
      for any new atom-type names introduced.
      
      Both pipeline entry points produce this same struct:
      
      * **AtomArray / SMILES path** — :func:`tmol.ligand.prepare_single_ligand`
        types the (already protonated, already charged) SMILES-derived molecule,
        builds the residue, and extracts cartbonded params, returning one
        ``LigandPreparation`` per ligand.
      * **Params-file path** — :func:`tmol.ligand.params_file.load_params_file`
        parses a ``.tmol`` YAML and returns ``list[LigandPreparation]``
        describing the residues defined in that file.
      
      Either list is then handed to :func:`inject_ligand_preparations`,
      the single chokepoint that extends the ``ParameterDatabase``. Tests
      can equally roundtrip ``AtomArray → LigandPreparation → .tmol →
      LigandPreparation`` and expect bit-equivalent injection.
      

   .. py:attribute:: residue_type
      :type:  tmol.database.chemical.RawResidueType


   .. py:attribute:: partial_charges
      :type:  dict[str, float]


   .. py:attribute:: cartbonded_params
      :type:  tmol.database.scoring.cartbonded.CartRes


   .. py:attribute:: atom_type_elements
      :type:  Optional[dict[str, str]]
      :value: None



.. py:function:: inject_ligand_preparations(param_db: tmol.database.ParameterDatabase, preparations: list[LigandPreparation], *, strict_atom_types: bool = False) -> tmol.database.ParameterDatabase

   .. rubric:: Docstring

   .. code-block:: text

      Inject a batch of ``LigandPreparation`` records into a database.
      
      The single chokepoint both pipeline paths use — given a list of
      prepared ligands (regardless of whether they came from a
      ``.tmol`` file or an AtomArray), this function aggregates their
      residue types, atom types, charges, and cartbonded params and
      evolves the input ``ParameterDatabase`` exactly once via
      :func:`tmol.database.inject_residue_params`.
      
      Residues whose name already exists in ``param_db`` are silently
      skipped so repeat injection is idempotent.
      
      :param param_db: Base database (not modified).
      :param preparations: One ``LigandPreparation`` per ligand to register.
      :param strict_atom_types: If True, raise when an atom type's element
                                cannot be resolved from any preparation's
                                ``atom_type_elements`` — otherwise fall back to a name-based
                                heuristic and emit a warning.
      
      :returns: A new frozen ``ParameterDatabase`` extended with all provided
                preparations.
      

.. py:function:: rebuild_canonical_ordering(param_db: tmol.database.ParameterDatabase) -> tmol.io.canonical_ordering.CanonicalOrdering

   .. rubric:: Docstring

   .. code-block:: text

      Build a new CanonicalOrdering from a (possibly extended) ParameterDatabase.
      

