tmol.chemical.ideal_coords
==========================

.. py:module:: tmol.chemical.ideal_coords


Functions
---------

.. autoapisummary::

   tmol.chemical.ideal_coords.eye4
   tmol.chemical.ideal_coords.normalize
   tmol.chemical.ideal_coords.frame_from_coords
   tmol.chemical.ideal_coords.rot_x
   tmol.chemical.ideal_coords.rot_z
   tmol.chemical.ideal_coords.trans_z
   tmol.chemical.ideal_coords.build_coords_from_icoors
   tmol.chemical.ideal_coords.build_ideal_coords


Module Contents
---------------

.. py:function:: eye4()

   .. rubric:: Docstring

   .. code-block:: text

      Create the identity homogeneous transform
      Only necessary because numpy.eye(4, dtype=numpy.float32)
      is strangely unsupported in numpy
      

.. py:function:: normalize(v)

.. py:function:: frame_from_coords(p1, p2, p3)

.. py:function:: rot_x(rot)

.. py:function:: rot_z(rot)

.. py:function:: trans_z(trans)

.. py:function:: build_coords_from_icoors(icoors_ancestors, icoors_geom)

.. py:function:: build_ideal_coords(restype: RefinedResidueType)

