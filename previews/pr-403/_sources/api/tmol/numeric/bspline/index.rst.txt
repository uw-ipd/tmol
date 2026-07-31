tmol.numeric.bspline
====================

.. py:module:: tmol.numeric.bspline

.. rubric:: Module docstring

.. code-block:: text

   N-dimensional B-spline interpolation with periodic boundary conditions.
   
   Reference:
       Thévenaz, Philippe, Thierry Blu, and Michael Unser.
       "Interpolation revisited [medical images application]."
       IEEE Transactions on medical imaging 19.7 (2000): 739-758.
   
       http://bigwww.epfl.ch/publications/thevenaz0002.pdf
   
   B-splines can be used for interpolation, using only as much space
   as the original data, after the original data is processed to produce
   coefficients for the polynomials. When interpolating, B-splines read from
   as much memory as bicublic-spline interpolation does (e.g. 16 values
   in 2D, 64 values in 3D), but they read from a wider number of grid cells
   to do so, instead of making each grid cell contain more entries. For this
   reason, the memory footprint for B-splines is substantially lower than
   that for bicuplic spline interpolation. (Catmull-Rom splines are similarly
   low-memory overhead, but do not fit the data as cleanly).
   
   To use these B-splines, construct a BSplineInterpolation object using the
   ``from_coordinates`` function, passing in the tensor of coordinates that
   should be interpolated, indicating the degree of the spline (3 for
   the equivalent of bicubic spline interpolation), and indicating the
   number of dimensions in the coordinate tensor that are indexing rather
   than interpolating. (Indexing dimensions must appear first). Indexing
   dimensions allow stacks of coordinate tensors to be interpolated
   simultaneously as might be useful, e.g., if one had 20 different 36x36
   tables as one does when computing the Ramachandran potential.
   
   Interpolation is performed where the input X values must be in the range of
   (0, len(X_i)] for dimension i -- if, e.g., you are interpolating dihedrals
   in degrees with a 10 degree step size in the range (-180, 180], then add 180
   to the dihedral shifting to the range [0, 360) and then divide by 10 to produce
   an interpolation value in the range (0, 36]. Another way to say this is that
   the code assumes a uniform unit distance between interpolation points.
   


Classes
-------

.. autoapisummary::

   tmol.numeric.bspline.BSplineInterpolation


Module Contents
---------------

.. py:class:: BSplineInterpolation

   .. rubric:: Docstring

   .. code-block:: text

      Class for performing bspline interpolation with periodic boundary conditions.
      
      Construct an instance of this class using the ``from_coordinates`` function, handing
      it a (possibly stacked) table of coordinates that should be interpolated by the
      splines, the degree of the spline that should be constructed, and (optionally)
      the number of dimensions in the coordinates tensor that are "indexing dimensions"
      and not interpolation dimensions. The indexing dimensions should appear as the
      most significant dimensions and the interpolating dimensions should appear as the
      least significant dimensions.
      
      Once constructed, the ``interpolate`` method can be given a tensor of coordinates
      ``X`` (and if the original coordinate tensor had indexing dimensions, a tensor of
      indices ``Y``) to produce a tensor of interpolated values.
      

   .. py:attribute:: coeffs
      :type:  tmol.types.torch.Tensor[torch.float]


   .. py:attribute:: n_interp_dims
      :type:  int


   .. py:method:: from_coordinates(coords: tmol.types.torch.Tensor[torch.float])
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Construct a BSplineInterpolation instance from  the input coordinates
         (i.e. the data to be interpolated)
         
         This code handles splines of 2-4 dimensions
         


   .. py:method:: interpolate(X: tmol.types.torch.Tensor[torch.float][Ellipsis]) -> tmol.types.torch.Tensor[torch.float]

      .. rubric:: Docstring

      .. code-block:: text

         B-spline interpolation function
         
         X should be a two dimensional tensor of size [ n_points, n_dims ]
         the result will be a one dimensional tensor of size [ n_points ].
         
         n_points represents the number of points in the n-dimensional space that is
         being interpolated (n-dimensional == n_interp_dims-dimensional). The
         result returned by this is the interpolated value for each of the input points.
         


