tmol.utility.ndarray.common_operations
======================================

.. py:module:: tmol.utility.ndarray.common_operations


Functions
---------

.. autoapisummary::

   tmol.utility.ndarray.common_operations.exclusive_cumsum1d
   tmol.utility.ndarray.common_operations.exclusive_cumsum2d
   tmol.utility.ndarray.common_operations.invert_mapping


Module Contents
---------------

.. py:function:: exclusive_cumsum1d(inds: Union[tmol.types.array.NDArray[numpy.int32][:], tmol.types.array.NDArray[numpy.int64][:]]) -> Union[tmol.types.array.NDArray[numpy.int32][:], tmol.types.array.NDArray[numpy.int64][:]]

.. py:function:: exclusive_cumsum2d(inds: Union[tmol.types.array.NDArray[numpy.int32][:, :], tmol.types.array.NDArray[numpy.int64][:, :]]) -> Union[tmol.types.array.NDArray[numpy.int32][:, :], tmol.types.array.NDArray[numpy.int64][:, :]]

.. py:function:: invert_mapping(a_2_b: Union[tmol.types.array.NDArray[numpy.int32][:], tmol.types.array.NDArray[numpy.int64][:]], n_elements_b: Optional[int] = None, sentinel: Optional[int] = -1)

