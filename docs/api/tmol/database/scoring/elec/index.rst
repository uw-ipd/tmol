tmol.database.scoring.elec
==========================

.. py:module:: tmol.database.scoring.elec


Classes
-------

.. autoapisummary::

   tmol.database.scoring.elec.GlobalParams
   tmol.database.scoring.elec.CountPairReps
   tmol.database.scoring.elec.PartialCharges
   tmol.database.scoring.elec.ElecDatabase


Module Contents
---------------

.. py:class:: GlobalParams

   .. py:attribute:: elec_min_dis
      :type:  float


   .. py:attribute:: elec_max_dis
      :type:  float


   .. py:attribute:: elec_sigmoidal_die_D
      :type:  float


   .. py:attribute:: elec_sigmoidal_die_D0
      :type:  float


   .. py:attribute:: elec_sigmoidal_die_S
      :type:  float


.. py:class:: CountPairReps

   .. py:attribute:: res
      :type:  str


   .. py:attribute:: atm_inner
      :type:  str


   .. py:attribute:: atm_outer
      :type:  str


.. py:class:: PartialCharges

   .. py:attribute:: res
      :type:  str


   .. py:attribute:: atom
      :type:  str


   .. py:attribute:: charge
      :type:  float


.. py:class:: ElecDatabase

   .. py:attribute:: global_parameters
      :type:  GlobalParams


   .. py:attribute:: atom_cp_reps_parameters
      :type:  Tuple[CountPairReps, Ellipsis]


   .. py:attribute:: atom_charge_parameters
      :type:  Tuple[PartialCharges, Ellipsis]


   .. py:method:: from_file(path)
      :classmethod:



