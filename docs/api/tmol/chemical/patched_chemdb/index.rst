tmol.chemical.patched_chemdb
============================

.. py:module:: tmol.chemical.patched_chemdb


Classes
-------

.. autoapisummary::

   tmol.chemical.patched_chemdb.RestypeGraphBuilder
   tmol.chemical.patched_chemdb.PatchedChemicalDatabase


Functions
---------

.. autoapisummary::

   tmol.chemical.patched_chemdb.remove_atom
   tmol.chemical.patched_chemdb.modify_atom
   tmol.chemical.patched_chemdb.update_icoor
   tmol.chemical.patched_chemdb.get_modified_atoms
   tmol.chemical.patched_chemdb.validate_raw_residue
   tmol.chemical.patched_chemdb.validate_patch
   tmol.chemical.patched_chemdb.do_patch


Module Contents
---------------

.. py:class:: RestypeGraphBuilder(atomtypedict)

   .. py:attribute:: atomtypedict


   .. py:method:: from_raw_res(r)


.. py:function:: remove_atom(res, atom)

.. py:function:: modify_atom(res, atom)

.. py:function:: update_icoor(res, patch, atoms_remove, namemap)

.. py:function:: get_modified_atoms(patch)

.. py:function:: validate_raw_residue(res)

.. py:function:: validate_patch(patch)

   .. rubric:: Docstring

   .. code-block:: text

      Validate a given patch object or raise a RuntimeException
      

.. py:function:: do_patch(res, variant, resgraph, patchgraph, marked)

.. py:class:: PatchedChemicalDatabase

   .. py:attribute:: element_types
      :type:  Tuple[tmol.database.chemical.Element, Ellipsis]


   .. py:attribute:: atom_types
      :type:  Tuple[tmol.database.chemical.AtomType, Ellipsis]


   .. py:attribute:: residues
      :type:  Tuple[tmol.database.chemical.RawResidueType, Ellipsis]


   .. py:attribute:: variants
      :type:  Tuple[tmol.database.chemical.VariantType, Ellipsis]


   .. py:method:: from_chem_db(chemdb: tmol.database.chemical.ChemicalDatabase)
      :classmethod:



