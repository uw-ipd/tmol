tmol.utility.cumsum
===================

.. py:module:: tmol.utility.cumsum


Functions
---------

.. autoapisummary::

   tmol.utility.cumsum.exclusive_cumsum
   tmol.utility.cumsum.exclusive_cumsum1d
   tmol.utility.cumsum.exclusive_cumsum2d
   tmol.utility.cumsum.exclusive_cumsum2d_w_totals


Module Contents
---------------

.. py:function:: exclusive_cumsum(inds: tmol.types.array.NDArray[int][:]) -> tmol.types.array.NDArray[int][:]

   .. rubric:: Docstring

   .. code-block:: text

      Calculate exclusive cumulative sum over input array
      

.. py:function:: exclusive_cumsum1d(inds: Union[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int64][:]]) -> Union[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int64][:]]

.. py:function:: exclusive_cumsum2d(inds: Union[tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.int64][:, :]]) -> Union[tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.int64][:, :]]

.. py:function:: exclusive_cumsum2d_w_totals(inds: Union[tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.int64][:, :]]) -> Union[Tuple[tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.int32][:]], Union[tmol.types.torch.Tensor[torch.int64][:, :], tmol.types.torch.Tensor[torch.int64][:]]]

