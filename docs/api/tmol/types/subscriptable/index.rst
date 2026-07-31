tmol.types.subscriptable
========================

.. py:module:: tmol.types.subscriptable


Classes
-------

.. autoapisummary::

   tmol.types.subscriptable.SubscriptableType


Module Contents
---------------

.. py:class:: SubscriptableType

   Bases: :py:obj:`type`


   .. rubric:: Docstring

   .. code-block:: text

      This metaclass will allow a type to become subscriptable.
      
      >>> class SomeType(metaclass=SubscriptableType):
      ...     pass
      >>> SomeTypeSub = SomeType['some args']
      >>> SomeTypeSub.__args__
      'some args'
      >>> SomeTypeSub.__origin__.__name__
      'SomeType'
      

