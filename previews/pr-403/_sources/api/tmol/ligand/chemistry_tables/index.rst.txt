tmol.ligand.chemistry_tables
============================

.. py:module:: tmol.ligand.chemistry_tables

.. rubric:: Module docstring

.. code-block:: text

   Database-backed ligand chemistry lookup tables.
   
   These helpers derive atom-class sets and hbond metadata from the default
   chemical database so atom-type updates can be handled centrally in YAML.
   


Functions
---------

.. autoapisummary::

   tmol.ligand.chemistry_tables.get_hbond_properties
   tmol.ligand.chemistry_tables.get_polar_classes
   tmol.ligand.chemistry_tables.get_sp2_atom_types


Module Contents
---------------

.. py:function:: get_hbond_properties() -> dict[str, dict[str, Any]]

   .. rubric:: Docstring

   .. code-block:: text

      Build hydrogen-bond related properties by atom-type name.
      
      :returns: Mapping from atom-type name to a compact property dictionary.
      

.. py:function:: get_polar_classes() -> frozenset[str]

   .. rubric:: Docstring

   .. code-block:: text

      Return configured legacy polar classes available in the database.
      
      :returns: Set of polar atom-class names that exist in the current database.
      

.. py:function:: get_sp2_atom_types() -> frozenset[str]

   .. rubric:: Docstring

   .. code-block:: text

      Collect atom-type names treated as sp2-like for typing helpers.
      
      :returns: Set of atom-type names considered sp2-like.
      

