tmol.database.scoring.disulfide
===============================

.. py:module:: tmol.database.scoring.disulfide


Classes
-------

.. autoapisummary::

   tmol.database.scoring.disulfide.DisulfideGlobalParameters
   tmol.database.scoring.disulfide.DisulfideDatabase


Module Contents
---------------

.. py:class:: DisulfideGlobalParameters

   .. py:attribute:: d_location
      :type:  float


   .. py:attribute:: d_scale
      :type:  float


   .. py:attribute:: d_shape
      :type:  float


   .. py:attribute:: a_logA
      :type:  float


   .. py:attribute:: a_kappa
      :type:  float


   .. py:attribute:: a_mu
      :type:  float


   .. py:attribute:: dss_logA1
      :type:  float


   .. py:attribute:: dss_kappa1
      :type:  float


   .. py:attribute:: dss_mu1
      :type:  float


   .. py:attribute:: dss_logA2
      :type:  float


   .. py:attribute:: dss_kappa2
      :type:  float


   .. py:attribute:: dss_mu2
      :type:  float


   .. py:attribute:: dcs_logA1
      :type:  float


   .. py:attribute:: dcs_mu1
      :type:  float


   .. py:attribute:: dcs_kappa1
      :type:  float


   .. py:attribute:: dcs_logA2
      :type:  float


   .. py:attribute:: dcs_mu2
      :type:  float


   .. py:attribute:: dcs_kappa2
      :type:  float


   .. py:attribute:: dcs_logA3
      :type:  float


   .. py:attribute:: dcs_mu3
      :type:  float


   .. py:attribute:: dcs_kappa3
      :type:  float


   .. py:attribute:: wt_dih_ss
      :type:  float


   .. py:attribute:: wt_dih_cs
      :type:  float


   .. py:attribute:: wt_ang
      :type:  float


   .. py:attribute:: wt_len
      :type:  float


   .. py:attribute:: shift
      :type:  float


.. py:class:: DisulfideDatabase

   .. py:attribute:: global_parameters
      :type:  DisulfideGlobalParameters


   .. py:method:: from_file(path)
      :classmethod:



