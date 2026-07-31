tmol.utility.categorical
========================

.. py:module:: tmol.utility.categorical

.. rubric:: Module docstring

.. code-block:: text

   Support for `enum`/`pandas.Categorical <pandas:api.categorical>` interconversion.
   


Functions
---------

.. autoapisummary::

   tmol.utility.categorical.enum_val_catdtype
   tmol.utility.categorical.enum_name_catdtype
   tmol.utility.categorical.vals_to_val_cat
   tmol.utility.categorical.vals_to_name_cat
   tmol.utility.categorical.names_to_name_cat
   tmol.utility.categorical.names_to_val_cat


Module Contents
---------------

.. py:function:: enum_val_catdtype(enum_type: enum.Enum) -> pandas.api.types.CategoricalDtype

   .. rubric:: Docstring

   .. code-block:: text

      Generate categorial dtype convering enumeratation values.
      

.. py:function:: enum_name_catdtype(enum_type: enum.Enum) -> pandas.api.types.CategoricalDtype

   .. rubric:: Docstring

   .. code-block:: text

      Generate categorial dtype convering enumeratation member names.
      

.. py:function:: vals_to_val_cat(enum_type: enum.Enum, values) -> pandas.Categorical

   .. rubric:: Docstring

   .. code-block:: text

      Convert enum values to a categorial.
      

.. py:function:: vals_to_name_cat(enum_type: enum.Enum, values) -> pandas.Categorical

   .. rubric:: Docstring

   .. code-block:: text

      Convert enum values to a categorial of member names.
      

.. py:function:: names_to_name_cat(enum_type: enum.Enum, values) -> pandas.Categorical

   .. rubric:: Docstring

   .. code-block:: text

      Convert enum names to a categorial.
      

.. py:function:: names_to_val_cat(enum_type: enum.Enum, values) -> pandas.Categorical

   .. rubric:: Docstring

   .. code-block:: text

      Convert enum names to a categorial of enum values.
      

