tmol.utility.mixins
===================

.. py:module:: tmol.utility.mixins

.. rubric:: Module docstring

.. code-block:: text

   Utility functions to support mixin classes.
   


Attributes
----------

.. autoapisummary::

   tmol.utility.mixins.QualifiedName


Functions
---------

.. autoapisummary::

   tmol.utility.mixins.qualified_name
   tmol.utility.mixins.gather_superclass_properies
   tmol.utility.mixins.cooperative_superclass_factory


Module Contents
---------------

.. py:data:: QualifiedName

.. py:function:: qualified_name(obj: Union[Type, Callable]) -> QualifiedName

   .. rubric:: Docstring

   .. code-block:: text

      The fully qualified <module>.<name> for a class/function.
      

.. py:function:: gather_superclass_properies(obj: Any, property_name: str) -> Dict[QualifiedName, Any]

   .. rubric:: Docstring

   .. code-block:: text

      Gather property values from all base classes of an object.
      
      Traverses the object's __mro__ searching for the given property name. The
      property fget is invoked for *every* property definition and the property
      values are returned as a mapping from class name to property value.
      

.. py:function:: cooperative_superclass_factory(cls, factory_func_name, *args, **kwargs)

   .. rubric:: Docstring

   .. code-block:: text

      Gather class factory components from subclasses and create object.
      
      Traverses a class __mro__ in *reverse* order accumulating __init__
      parameters via calls to class-level factory functions. Each factory
      function generates __init__ parameters by inspecting the factory function
      args, kwargs & current parameters and returning a parameter dict. Params
      are accumulated from factories via `dict.update`, making the partial
      results of param generation availabe to up-MRO factory functions.
      
      Note that the factory functions receive all current params as kwargs. In
      cases when the total class MRO is unknown a factory function should accept,
      and likely ignore, unknown kwargs. (Eg. ``def factory(cls, known, **_)``)
      
      Returns a dict of kwarg params accumulated from factory functions.
      

