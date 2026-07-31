tmol.utility.attr
=================

.. py:module:: tmol.utility.attr

.. rubric:: Module docstring

.. code-block:: text

   Mixin components for attrs-based classes.
   


Classes
-------

.. autoapisummary::

   tmol.utility.attr.AttrMapping
   tmol.utility.attr.AttrMutableMapping


Module Contents
---------------

.. py:class:: AttrMapping

   Bases: :py:obj:`collections.abc.Mapping`


   .. rubric:: Docstring

   .. code-block:: text

      Mixin adding Mapping interface to attr classes.
      

.. py:class:: AttrMutableMapping

   Bases: :py:obj:`AttrMapping`, :py:obj:`collections.abc.MutableMapping`


   .. rubric:: Docstring

   .. code-block:: text

      Mixin adding a subset of the mutable mapping interface to attr classes.
      
      As the keys of an attrs-based class are based on defined properties, this mixin
      does *not* support ``__delitem__``-based components of the MutableMapping interface,
      (eg. ``m.pop(key)``, ``del m[key]``, ...)
      

