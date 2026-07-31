tmol.utility.units
==================

.. py:module:: tmol.utility.units

.. rubric:: Module docstring

.. code-block:: text

   `pint <pint:index>`-based unit support functions.
   


Attributes
----------

.. autoapisummary::

   tmol.utility.units.ureg
   tmol.utility.units.u
   tmol.utility.units.Angle
   tmol.utility.units.BondAngle
   tmol.utility.units.DihedralAngle


Functions
---------

.. autoapisummary::

   tmol.utility.units.parse_angle
   tmol.utility.units.parse_bond_angle
   tmol.utility.units.parse_dihedral_angle


Module Contents
---------------

.. py:data:: ureg

.. py:data:: u

.. py:function:: parse_angle(angle: Union[float, str], lim: Tuple[float, float] = (-2 * math.pi, 2 * math.pi)) -> float

   .. rubric:: Docstring

   .. code-block:: text

      Parse an angle via :doc:`pint <pint:index>` and convert to radians.
      
      :param angle: Unit-qualified angle or float value in radians.
      :param lim: Raise ValueError if outside [min, max] range in radians.
      
      :returns: Angle in radians.
      

.. py:function:: parse_bond_angle(v: Union[float, str]) -> float

   .. rubric:: Docstring

   .. code-block:: text

      Parse a bond angle on the range [0, pi) via pint.
      

.. py:function:: parse_dihedral_angle(v) -> float

   .. rubric:: Docstring

   .. code-block:: text

      Parse a dihedral angle on the range [-pi, pi) via pint.
      

.. py:data:: Angle

.. py:data:: BondAngle

.. py:data:: DihedralAngle

