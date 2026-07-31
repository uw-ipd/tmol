tmol.types.tensor
=================

.. py:module:: tmol.types.tensor

.. rubric:: Module docstring

.. code-block:: text

   Type annotations for multidimensional tensors.
   


Classes
-------

.. autoapisummary::

   tmol.types.tensor.TensorGroup


Functions
---------

.. autoapisummary::

   tmol.types.tensor.cat


Module Contents
---------------

.. py:class:: TensorGroup

   .. py:method:: reshape(*shape)


   .. py:property:: shape


   .. py:method:: full(shape, fill_value, **kwargs)
      :classmethod:



   .. py:method:: zeros(shape, **kwargs)
      :classmethod:



   .. py:method:: ones(shape, **kwargs)
      :classmethod:



   .. py:method:: empty(shape, **kwargs)
      :classmethod:



   .. py:method:: to(*args, **kwargs)

      .. rubric:: Docstring

      .. code-block:: text

         Perform dtype/device conversion for all subtensors.
         
         Note that this may be an invalid operations if the TensorGroup contains
         heterogenous tensor dtypes.
         
         Performs Tensor dtype and/or device conversion. A :class:`torch.dtype`
         and :class:`torch.device` are inferred from the arguments of
         ``self.to(*args, **kwargs)``.
         
         If all subtensors already have the correct dtype and device then
         ``self`` is returned.
         


.. py:function:: cat(seq, dim=0, out=None)

