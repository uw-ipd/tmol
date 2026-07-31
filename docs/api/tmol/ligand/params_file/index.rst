tmol.ligand.params_file
=======================

.. py:module:: tmol.ligand.params_file

.. rubric:: Module docstring

.. code-block:: text

   tmol YAML params file format for ligand residue types.
   
   Provides load/write/inject functions for a unified YAML format that
   bundles residue type definitions, cartbonded parameters, and electrostatic
   charges in a single file.  The top-level shape mirrors ``ParameterDatabase``:
   
       version: "1.0"
       chemical:
         residues:
           - name: LIG
             base_name: LIG
             atoms: [...]
             bonds: [...]
             icoors: [...]
             properties: {...}
             # atom_aliases / chi_samples / default_jump_connection_atom optional
       elec:
         atom_charge_parameters:
           - {res: LIG, atom: C1, charge: 0.123}
       cartbonded:
         residue_params:
           LIG:
             length_parameters: [...]
             angle_parameters: [...]
             torsion_parameters: [...]
             improper_parameters: [...]
             hxltorsion_parameters: []
   
   Each subsection's schema matches the corresponding canonical database
   YAML so entries can be copy-pasted between params files and
   ``chemical.yaml`` / ``cartbonded.yaml`` / ``elec.yaml``.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.params_file.logger
   tmol.ligand.params_file.TMOL_FORMAT_VERSION


Functions
---------

.. autoapisummary::

   tmol.ligand.params_file.load_params_file
   tmol.ligand.params_file.inject_params_file
   tmol.ligand.params_file.inject_params_files


Module Contents
---------------

.. py:data:: logger

.. py:data:: TMOL_FORMAT_VERSION
   :type:  str
   :value: '1.0'


.. py:function:: load_params_file(path: str | pathlib.Path) -> list[tmol.ligand.registry.LigandPreparation]

   .. rubric:: Docstring

   .. code-block:: text

      Load a tmol params YAML file as a list of ``LigandPreparation``.
      
      The returned list is the same abstraction the AtomArray pipeline
      produces (see :func:`tmol.ligand.prepare_single_ligand`), so the
      caller can pass it directly to
      :func:`tmol.ligand.registry.inject_ligand_preparations` regardless
      of which input form (file or AtomArray) it came from.
      
      The ``.tmol`` schema is the nested
      ``chemical:`` / ``elec:`` / ``cartbonded:`` shape — files using the
      legacy flat schema (top-level ``residues:`` etc.) raise a
      ``ValueError`` pointing at the migration.
      

.. py:function:: inject_params_file(param_db: tmol.database.ParameterDatabase, path: str | pathlib.Path, *, strict_atom_types: bool = False) -> tmol.database.ParameterDatabase

   .. rubric:: Docstring

   .. code-block:: text

      Load a single ``.tmol`` file and inject it into a ParameterDatabase.
      

.. py:function:: inject_params_files(param_db: tmol.database.ParameterDatabase, paths: list[str | pathlib.Path], *, strict_atom_types: bool = False) -> tmol.database.ParameterDatabase

   .. rubric:: Docstring

   .. code-block:: text

      Load multiple ``.tmol`` files and inject them in one shot.
      

