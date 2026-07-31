tmol.types.shape
================

.. py:module:: tmol.types.shape

.. rubric:: Module docstring

.. code-block:: text

   Shape Specifications
   --------------------
   
   Shape specifications are intended to allow a reasonable description of common
   array shapes, with an emphasis on functionally relevant shape cases. This
   includes dimensionality, shape of specific dimensions, implied broadcastable
   dimensions, contiguous ordering (ie. C vs F ordering), and density.
   
   ``ndim`` and ``shape``
   ~~~~~~~~~~~~~~~~~~~~~~
   
   Basic dimensionality and shape requirements are specified via slices.
   Dimensions may be unconstrained or constrainted to a fixed shape.
   
   - ``[:]`` - ndim 1, any shape
   - ``[3]`` or ``[:3]`` - ndim 1, shape (3,)
   - ``[:,3]`` - ndim 2, shape (n,3)
   - ``[3,3]`` - ndim 2, shape (3,3)
   
   Broadcastable Dimensions
   ~~~~~~~~~~~~~~~~~~~~~~~~
   
   Optional dimensions are represented by an elipsis. This should generally be
   limited to *only* [implicitly
   broadcastable](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
   upper dimensions.
   
   - ``[...,:3]`` - ndim 1+, shape ([any]*, 3)
   - ``[...,:,:3]`` - ndim 2+, shape ([any]+, 3, 3)
   - ``[...,:3,:3]`` - ndim 2+, shape ([any]+, 3, 3)
   
   Stride and Contiguous Dimensions
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
   Memory layout constraints can be used to specify contiguous dimensions and
   their ordering. Dense dimensions are specified in the standard
   (inner|c|numpy|row-major) order or or in (outer|fortran|col-major) order.
   Any number of dimensions, starting from either ordering, can be specified as
   dense. Elements of dense dimensions are contiguous support a raveled view.
   
   The exact syntax for this dimension specification is unclear. The "inline",
   utilizing the slide step component:
   
   
   - ``[::1]`` - ndim 1, any shape, contiguous
   - ``[:,::1]`` - ndim 2, any shape, c-contiguous
   - ``[::1,:3]`` - ndim 2, shape (any, 3), f-contiguous
   - ``[:4:1j,:4:1]`` - ndim 2, shape (4,4), fully dense, c-contiguous
   - ``[:,:4:1j,:4:1]`` - ndim 3, shape (n, 4,4), 2-dense, c-contiguous
   
   Or a standard, ordering/density:
   
   - ``t[:].dense()`` - ndim 1, any shape, contiguous
   - ``t[:,:].order('c')`` - ndim 2, any shape, c-contiguous
   - ``t[:,:].dense().order('c')`` - ndim 2, any shape, fully dense, c-contiguous
   - ``t[:,3].dense(1).order('f')`` - ndim 2, shape (any, 3), f-contiguous
   - ``[:,4,4].dense(2).order('c')`` - ndim 3, shape (n, 4,4), 2-dense, c-contiguous
   


Classes
-------

.. autoapisummary::

   tmol.types.shape.Dim
   tmol.types.shape.Shape


Module Contents
---------------

.. py:class:: Dim

   .. py:attribute:: size


.. py:class:: Shape(dims)

   .. py:class:: Factory

      Bases: :py:obj:`object`



   .. py:attribute:: spec


   .. py:attribute:: dims


   .. py:method:: create(dims)
      :classmethod:



   .. py:method:: validate(shape)


