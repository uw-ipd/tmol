tmol.types.torch
================

.. py:module:: tmol.types.torch

.. rubric:: Module docstring

.. code-block:: text

   Tensor type attributes for torch arrays.
   


Classes
-------

.. autoapisummary::

   tmol.types.torch.Tensor


Functions
---------

.. autoapisummary::

   tmol.types.torch.torch_dtype
   tmol.types.torch.like_kwargs


Module Contents
---------------

.. py:function:: torch_dtype(dt)

   .. rubric:: Docstring

   .. code-block:: text

      Resolve a torch dtype via numpy's dtype parsing system.
      

.. py:function:: like_kwargs(t: torch.Tensor)

   .. rubric:: Docstring

   .. code-block:: text

      Extract kwargs args needed to initialize an identical tensor.
      

.. py:class:: Tensor

   Bases: :py:obj:`tmol.types.tensor._TensorType`


   .. py:method:: convert(value)
      :classmethod:



