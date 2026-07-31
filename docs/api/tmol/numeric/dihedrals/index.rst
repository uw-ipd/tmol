tmol.numeric.dihedrals
======================

.. py:module:: tmol.numeric.dihedrals


Attributes
----------

.. autoapisummary::

   tmol.numeric.dihedrals.Coord64Array
   tmol.numeric.dihedrals.Angles


Functions
---------

.. autoapisummary::

   tmol.numeric.dihedrals.coord_dihedrals


Module Contents
---------------

.. py:data:: Coord64Array

.. py:data:: Angles

.. py:function:: coord_dihedrals(a: Coord64Array, b: Coord64Array, c: Coord64Array, d: Coord64Array) -> Angles

   .. rubric:: Docstring

   .. code-block:: text

      Dihedral angle in [-pi, pi] over the planes defined by {a, b, c} & {b, c, d}.
      
      Calculate dihedral angle from four coordinate locations, using the
      "standard" torsion angle definition of two planes defined by the points
      {a, b, c} and {b, c, d}. For a four-atom bond definition, this corrosponds
      to rotation about the b-c bond.
      

