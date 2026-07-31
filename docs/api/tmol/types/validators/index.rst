tmol.types.validators
=====================

.. py:module:: tmol.types.validators

.. rubric:: Module docstring

.. code-block:: text

   Generic type validator functions.
   


Functions
---------

.. autoapisummary::

   tmol.types.validators.is_list_type
   tmol.types.validators.get_validator
   tmol.types.validators.validate_tuple
   tmol.types.validators.validate_list
   tmol.types.validators.validate_union
   tmol.types.validators.validate_isinstance
   tmol.types.validators.register_validator


Module Contents
---------------

.. py:function:: is_list_type(tp)

   .. rubric:: Docstring

   .. code-block:: text

      Test if the type is a generic list type, including subclasses excluding
      non-generic classes.
      Examples::
          is_list_type(int) == False
          is_list_type(list) == False
          is_list_type(List) == True
          is_list_type(List[int]) == True
          is_list_type(List[str, int]) == True
          class MyClass(List[str, int]):
              ...
          is_tuple_type(MyClass) == True
      For more general tests use issubclass(..., list), for more precise test
      (excluding subclasses) use::
          get_origin(tp) is list  # Tuple prior to Python 3.7
      

.. py:function:: get_validator(type_annotation)

.. py:function:: validate_tuple(tup, value)

.. py:function:: validate_list(lst, value)

   .. rubric:: Docstring

   .. code-block:: text

      Test if a given value matches the List type in the type hints:
      A list may either be of a uniform type, e.g. "List[int]", or may have
      no specified type, and thus be of any time, e.g. "List". In the first
      case, the single type may be a Union, e.g. "List[Union[int, str]]".
      
      validate_list(List[int], [5]) == True
      validate_list(List[int], [5, 4, 3]) == True
      validate_list(List[int], [5, "thumb"]) == False
      validate_list(List, 5) == False
      validate_list(List, []) == True
      validate_list(List, [5, "thumb"]) == True
      

.. py:function:: validate_union(union, value)

.. py:function:: validate_isinstance(type_annotation, value)

.. py:function:: register_validator(type_predicate, validator)

