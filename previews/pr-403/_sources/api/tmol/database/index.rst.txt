tmol.database
=============

.. py:module:: tmol.database


Submodules
----------

.. toctree::
   :maxdepth: 1

   /api/tmol/database/chemical/index
   /api/tmol/database/scoring/index


Classes
-------

.. autoapisummary::

   tmol.database.ParameterDatabase


Functions
---------

.. autoapisummary::

   tmol.database.inject_residue_params


Package Contents
----------------

.. py:class:: ParameterDatabase

   .. rubric:: Docstring

   .. code-block:: text

      Immutable chemical and scoring parameter container used by tmol.
      
      The process-global accessor ``get_default()`` returns a shared read-only
      instance.  To add ligand or custom residue data, use
      :func:`inject_residue_params` which returns a **new** database.
      

   .. py:method:: get_default() -> ParameterDatabase
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Return the process-global cached parameter database (read-only).
         


   .. py:attribute:: scoring
      :type:  scoring.ScoringDatabase


   .. py:attribute:: chemical
      :type:  tmol.chemical.patched_chemdb.PatchedChemicalDatabase


   .. py:method:: from_file(path: str) -> ParameterDatabase
      :classmethod:



   .. py:method:: create_stable_subset(desired_names: List[str], desired_variants: List[str]) -> ParameterDatabase

      .. rubric:: Docstring

      .. code-block:: text

         Create a ParameterDatabase representing a subset of the
         RefinedResidueTypes in this PD's PatchedChemicalDatabase from a list
         of RRT names and patched with the given variants (identified by their
         display names) where the order in which RRTs will appear in the subset
         will be stable over time (as long as this source PCD is only accumulating
         new RRTs over time and not losing the RRTs that it starts with).
         
         


.. py:function:: inject_residue_params(param_db: ParameterDatabase, residue_types: list[chemical.RawResidueType], atom_types: Optional[list[chemical.AtomType]] = None, partial_charges: Optional[Mapping[str, dict[str, float]]] = None, cartbonded_params: Optional[Mapping[str, scoring.cartbonded.CartRes]] = None) -> ParameterDatabase

   .. rubric:: Docstring

   .. code-block:: text

      Return a new ParameterDatabase with additional residue type data.
      
      This is the primary API for extending a database with ligand or custom
      residue types.  The input ``param_db`` is not modified.
      
      :param param_db: Base database to extend.
      :param residue_types: New RawResidueType entries to add.
      :param atom_types: Optional new AtomType entries (deduplicated by name).
      :param partial_charges: Per-residue charge dicts ``{res_name: {atom: charge}}``.
      :param cartbonded_params: Per-residue CartRes ``{res_name: CartRes}``.
      
      :returns: A new frozen ParameterDatabase with the additional data.
      

