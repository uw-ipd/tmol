tmol.utility.dicttoolz
======================

.. py:module:: tmol.utility.dicttoolz


Functions
---------

.. autoapisummary::

   tmol.utility.dicttoolz.items
   tmol.utility.dicttoolz.keys
   tmol.utility.dicttoolz.vals
   tmol.utility.dicttoolz.flat_items
   tmol.utility.dicttoolz.unflatten
   tmol.utility.dicttoolz.update_inplace


Module Contents
---------------

.. py:function:: items(d)

.. py:function:: keys(d)

.. py:function:: vals(d)

.. py:function:: flat_items(d)

   .. rubric:: Docstring

   .. code-block:: text

      Iterate items from potentially nested mapping.
      
      Iterates [(keys,...): value] items from a potentially nested mapping of
      mappings, where keys is a tuple of key-path leading to value. Traverses all
      collections.abc.Mapping subtypes.
      

.. py:function:: unflatten(keys_values, factory=dict)

   .. rubric:: Docstring

   .. code-block:: text

      Construct potentially-nested mapping from (keys, value) items.
      
      Construct a potentially-nested mapping from a flat iterator of (keys,
      value) pairs. This is functionally equivalent to reducing items via
      assoc_in.
      

.. py:function:: update_inplace(d, keys, func, default=None, factory=dict)

   .. rubric:: Docstring

   .. code-block:: text

      Update value in a (potentially) nested dictionary inplace
      
      inputs:
      d - dictionary on which to operate
      keys - list or tuple giving the location of the value to be changed in d
      func - function to operate on that value
      
      If keys == [k0,..,kX] and d[k0]..[kX] == v, update_inplace updates the
      original dictionary with v replaced by func(v).
      
      If k0 is not a key in d, update_inplace creates nested dictionaries to the
      depth specified by the keys, with the innermost value set to func(default).
      
      Returns d.
      

