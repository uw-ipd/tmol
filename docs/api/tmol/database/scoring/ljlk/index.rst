tmol.database.scoring.ljlk
==========================

.. py:module:: tmol.database.scoring.ljlk


Classes
-------

.. autoapisummary::

   tmol.database.scoring.ljlk.LJLKGlobalParameters
   tmol.database.scoring.ljlk.LJLKAtomTypeParameters
   tmol.database.scoring.ljlk.LJLKDatabase


Module Contents
---------------

.. py:class:: LJLKGlobalParameters

   .. py:attribute:: max_dis
      :type:  float


   .. py:attribute:: lj_hbond_OH_donor_dis
      :type:  float


   .. py:attribute:: lj_hbond_dis
      :type:  float


   .. py:attribute:: lj_hbond_hdis
      :type:  float


   .. py:attribute:: lj_dlin_sigma_factor
      :type:  float


   .. py:attribute:: lj_dlin_sigma_factor_soft
      :type:  float


   .. py:attribute:: lk_min_dis2sigma
      :type:  float


   .. py:attribute:: lkb_water_dist
      :type:  float


   .. py:attribute:: lkb_water_angle_sp2
      :type:  tmol.utility.units.Angle


   .. py:attribute:: lkb_water_angle_sp3
      :type:  tmol.utility.units.Angle


   .. py:attribute:: lkb_water_angle_ring
      :type:  tmol.utility.units.Angle


   .. py:attribute:: lkb_water_tors_sp2
      :type:  List[tmol.utility.units.Angle]


   .. py:attribute:: lkb_water_tors_sp3
      :type:  List[tmol.utility.units.Angle]


   .. py:attribute:: lkb_water_tors_ring
      :type:  List[tmol.utility.units.Angle]


.. py:class:: LJLKAtomTypeParameters

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: lj_radius
      :type:  float


   .. py:attribute:: lj_wdepth
      :type:  float


   .. py:attribute:: lk_dgfree
      :type:  float


   .. py:attribute:: lk_lambda
      :type:  float


   .. py:attribute:: lk_volume
      :type:  float


.. py:class:: LJLKDatabase

   .. py:attribute:: global_parameters
      :type:  LJLKGlobalParameters


   .. py:attribute:: atom_type_parameters
      :type:  Tuple[LJLKAtomTypeParameters, Ellipsis]


   .. py:method:: from_file(path)
      :classmethod:



