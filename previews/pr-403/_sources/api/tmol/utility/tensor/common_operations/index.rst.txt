tmol.utility.tensor.common_operations
=====================================

.. py:module:: tmol.utility.tensor.common_operations


Functions
---------

.. autoapisummary::

   tmol.utility.tensor.common_operations.stretch
   tmol.utility.tensor.common_operations.stretch2
   tmol.utility.tensor.common_operations.exclusive_cumsum1d
   tmol.utility.tensor.common_operations.exclusive_cumsum2d
   tmol.utility.tensor.common_operations.exclusive_cumsum2d_and_totals
   tmol.utility.tensor.common_operations.print_row_numbered_tensor
   tmol.utility.tensor.common_operations.nplus1d_tensor_from_list
   tmol.utility.tensor.common_operations.cat_differently_sized_tensors
   tmol.utility.tensor.common_operations.join_tensors_and_report_real_entries
   tmol.utility.tensor.common_operations.invert_mapping


Module Contents
---------------

.. py:function:: stretch(t: Union[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int64][:]], count)

   .. rubric:: Docstring

   .. code-block:: text

      take an input tensor and "repeat" each element count times.
      stretch(tensor([0, 1, 2, 3]), 3) returns:
           tensor([0 0 0 1 1 1 2 2 2 3 3 3]
      this is equivalent to numpy's repeat
      

.. py:function:: stretch2(t: Union[tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.int64][:, :]], count)

   .. rubric:: Docstring

   .. code-block:: text

      take an input 2D tensor and "repeat" each element count times.
      stretch2(tensor([[0, 1, 2, 3], [4, 5, 6, 7]]), 3) returns:
           tensor([[0 0 0 1 1 1 2 2 2 3 3 3],[4 4 4 5 5 5 6 6 6 7 7 7]])
      this is equivalent to numpy's repeat
      

.. py:function:: exclusive_cumsum1d(inds: Union[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int64][:]]) -> Union[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int64][:]]

.. py:function:: exclusive_cumsum2d(inds: Union[tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.int64][:, :]]) -> Union[tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.int64][:, :]]

.. py:function:: exclusive_cumsum2d_and_totals(inds: Union[tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.int64][:, :]]) -> Union[Tuple[tmol.types.torch.Tensor[torch.int32][:, :], tmol.types.torch.Tensor[torch.int32][:]], Tuple[tmol.types.torch.Tensor[torch.int64][:, :], tmol.types.torch.Tensor[torch.int64][:]]]

.. py:function:: print_row_numbered_tensor(tensor)

.. py:function:: nplus1d_tensor_from_list(tensors: List)

.. py:function:: cat_differently_sized_tensors(tensors: List)

.. py:function:: join_tensors_and_report_real_entries(tensors: List, sentinel: int = -1)

   .. rubric:: Docstring

   .. code-block:: text

      Concatenate a bunch of N-dimensional tensors into a single N+1-D tensor
      and report which elements out of the new tensor are real.
      The tensors may have different sizes for dimension 0 but should have the
      same size for all other dimensions. They must all have the same
      dtype and live on the same device.
      

.. py:function:: invert_mapping(a_2_b: Union[tmol.types.torch.Tensor[torch.int32][:], tmol.types.torch.Tensor[torch.int64][:]], n_elements_b: Optional[int] = None, sentinel: Optional[int] = -1)

