tmol.types.array
================

.. py:module:: tmol.types.array

.. rubric:: Module docstring

.. code-block:: text

   Tensor type attributes for numpy arrays.
   


Classes
-------

.. autoapisummary::

   tmol.types.array.Casting
   tmol.types.array.NDArray


Module Contents
---------------

.. py:class:: Casting(*args, **kwds)

   Bases: :py:obj:`enum.Enum`


   .. rubric:: Docstring

   .. code-block:: text

      Casting specifications for array types, see ndarray.astype.
      

   .. py:attribute:: no
      :value: 'no'



   .. py:attribute:: equiv
      :value: 'equiv'



   .. py:attribute:: safe
      :value: 'safe'



   .. py:attribute:: same_kind
      :value: 'same_kind'



   .. py:attribute:: unsafe
      :value: 'unsafe'



.. py:class:: NDArray

   Bases: :py:obj:`tmol.types.tensor._TensorType`


   .. py:attribute:: casting
      :type:  Casting


   .. py:method:: convert(value)
      :classmethod:



