tmol.numeric.interpolation.cubic_hermite_polynomial
===================================================

.. py:module:: tmol.numeric.interpolation.cubic_hermite_polynomial


Functions
---------

.. autoapisummary::

   tmol.numeric.interpolation.cubic_hermite_polynomial.interpolate_t
   tmol.numeric.interpolation.cubic_hermite_polynomial.interpolate
   tmol.numeric.interpolation.cubic_hermite_polynomial.interpolate_dt
   tmol.numeric.interpolation.cubic_hermite_polynomial.interpolate_dx
   tmol.numeric.interpolation.cubic_hermite_polynomial.interpolate_to_zero_t
   tmol.numeric.interpolation.cubic_hermite_polynomial.interpolate_to_zero
   tmol.numeric.interpolation.cubic_hermite_polynomial.interpolate_to_zero_dt
   tmol.numeric.interpolation.cubic_hermite_polynomial.interpolate_to_zero_dx


Module Contents
---------------

.. py:function:: interpolate_t(t, p0, dp0, p1, dp1)

   .. rubric:: Docstring

   .. code-block:: text

      Cubic interpolation of p on t in [0, 1].
      

.. py:function:: interpolate(x, x0, p0, dpdx0, x1, p1, dpdx1)

   .. rubric:: Docstring

   .. code-block:: text

      Cubic interpolation of p on x in [x0, x1].
      

.. py:function:: interpolate_dt(t, p0, dp0, p1, dp1)

   .. rubric:: Docstring

   .. code-block:: text

      Cubic interpolation of dp/dt on t in [0, 1].
      

.. py:function:: interpolate_dx(x, x0, p0, dpdx0, x1, p1, dpdx1)

   .. rubric:: Docstring

   .. code-block:: text

      Cubic interpolation of dp/dx on x in [x0, x1].
      

.. py:function:: interpolate_to_zero_t(t, p0, dp0)

   .. rubric:: Docstring

   .. code-block:: text

      Cubic interpolation of p on t in [0, 1] to (p1, dp1) == 0.
      

.. py:function:: interpolate_to_zero(x, x0, p0, dpdx0, x1)

   .. rubric:: Docstring

   .. code-block:: text

      Cubic interpolation of p on x in [x0, x1] to (p1, dpdx1) == 0 at x1.
      

.. py:function:: interpolate_to_zero_dt(t, p0, dp0)

   .. rubric:: Docstring

   .. code-block:: text

      Cubic interpolation of dp/dt on t in [0, 1] to (p1, dp1) == 0.
      

.. py:function:: interpolate_to_zero_dx(x, x0, p0, dpdx0, x1)

   .. rubric:: Docstring

   .. code-block:: text

      Cubic interpolation of dp/dx on x in [x0, x1] to (p1, dpdx1) == 0 at x1.
      

