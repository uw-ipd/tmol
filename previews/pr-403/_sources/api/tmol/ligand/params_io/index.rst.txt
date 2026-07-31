tmol.ligand.params_io
=====================

.. py:module:: tmol.ligand.params_io

.. rubric:: Module docstring

.. code-block:: text

   Read and write ligand params files (Rosetta ``.params`` and tmol ``.tmol``).
   
   Single home for ligand params I/O. :func:`write_params_file` serializes a
   :class:`~tmol.ligand.registry.LigandPreparation` to either format; the Rosetta
   reader :func:`read_params_file` and (re-exported) tmol reader cover the inputs.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.params_io.logger


Functions
---------

.. autoapisummary::

   tmol.ligand.params_io.read_params_file
   tmol.ligand.params_io.write_params_file
   tmol.ligand.params_io.write_params_from_mol2


Module Contents
---------------

.. py:data:: logger

.. py:function:: read_params_file(path: str | pathlib.Path) -> tmol.database.chemical.RawResidueType

   .. rubric:: Docstring

   .. code-block:: text

      Read a Rosetta .params file into a RawResidueType.
      
      Parses ATOM, BOND, ICOOR_INTERNAL, and NBR_ATOM records. Other
      records are silently ignored.
      
      :param path: Path to the .params file.
      
      :returns: A RawResidueType populated from the params file.
      

.. py:function:: write_params_file(preparation: LigandPreparation | list[LigandPreparation], path: str | pathlib.Path, format: str = 'rosetta') -> None

   .. rubric:: Docstring

   .. code-block:: text

      Write a ligand ``LigandPreparation`` as a Rosetta ``.params`` or tmol ``.tmol``.
      
      :param preparation: A :class:`~tmol.ligand.registry.LigandPreparation` (its
                          ``residue_type`` / ``partial_charges`` / ``cartbonded_params`` are
                          used), or a list of them.
      :param path: Output path. Its meaning depends on the format and whether a list
                   was passed:
      
                   * single preparation -> ``path`` is the output file (either format);
                   * ``"rosetta"`` + list -> ``path`` is a **directory**; each
                     preparation is written to ``<path>/<residue_type.name>.params``
                     (a ``.params`` holds a single residue);
                   * ``"tmol"`` (single or list) -> ``path`` is a single file holding
                     all residues.
      :param format: ``"rosetta"`` (classic Rosetta ``.params``) or ``"tmol"``
                     (tmol YAML ``.tmol``).
      

.. py:function:: write_params_from_mol2(mol2_path: str | pathlib.Path, out_path: str | pathlib.Path, *, res_name: str | None = None, sample_proton_chi: bool = True, format: str = 'rosetta') -> None

   .. rubric:: Docstring

   .. code-block:: text

      Build params from a mol2 file and write Rosetta ``.params`` or tmol ``.tmol``.
      
      :param mol2_path: Input Tripos mol2 (names, coords, charges preserved verbatim).
      :param out_path: Output file path (see :func:`write_params_file`).
      :param res_name: Optional residue name override.
      :param sample_proton_chi: Whether to emit PROTON_CHI samples.
      :param format: ``"rosetta"`` or ``"tmol"``.
      

