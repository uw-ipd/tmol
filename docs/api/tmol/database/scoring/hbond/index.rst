tmol.database.scoring.hbond
===========================

.. py:module:: tmol.database.scoring.hbond


Classes
-------

.. autoapisummary::

   tmol.database.scoring.hbond.GlobalParams
   tmol.database.scoring.hbond.DonorAtomType
   tmol.database.scoring.hbond.AcceptorAtomType
   tmol.database.scoring.hbond.DonorTypeParam
   tmol.database.scoring.hbond.AcceptorTypeParam
   tmol.database.scoring.hbond.PolynomialParameters
   tmol.database.scoring.hbond.PairParameters
   tmol.database.scoring.hbond.HBondDatabaseRaw
   tmol.database.scoring.hbond.HBondDatabase


Module Contents
---------------

.. py:class:: GlobalParams

   .. py:attribute:: hb_sp2_range_span
      :type:  float


   .. py:attribute:: hb_sp2_BAH180_rise
      :type:  float


   .. py:attribute:: hb_sp2_outer_width
      :type:  float


   .. py:attribute:: hb_sp3_softmax_fade
      :type:  float


   .. py:attribute:: threshold_distance
      :type:  float


.. py:class:: DonorAtomType

   .. py:attribute:: d
      :type:  str


   .. py:attribute:: donor_type
      :type:  str


.. py:class:: AcceptorAtomType

   .. py:attribute:: a
      :type:  str


   .. py:attribute:: acceptor_type
      :type:  str


.. py:class:: DonorTypeParam

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: weight
      :type:  float


.. py:class:: AcceptorTypeParam

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: weight
      :type:  float


.. py:class:: PolynomialParameters

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: dimension
      :type:  str


   .. py:attribute:: xmin
      :type:  float


   .. py:attribute:: xmax
      :type:  float


   .. py:attribute:: min_val
      :type:  float


   .. py:attribute:: max_val
      :type:  float


   .. py:attribute:: degree
      :type:  int


   .. py:attribute:: c_0
      :type:  float
      :value: 0.0



   .. py:attribute:: c_1
      :type:  float
      :value: 0.0



   .. py:attribute:: c_2
      :type:  float
      :value: 0.0



   .. py:attribute:: c_3
      :type:  float
      :value: 0.0



   .. py:attribute:: c_4
      :type:  float
      :value: 0.0



   .. py:attribute:: c_5
      :type:  float
      :value: 0.0



   .. py:attribute:: c_6
      :type:  float
      :value: 0.0



   .. py:attribute:: c_7
      :type:  float
      :value: 0.0



   .. py:attribute:: c_8
      :type:  float
      :value: 0.0



   .. py:attribute:: c_9
      :type:  float
      :value: 0.0



   .. py:attribute:: c_10
      :type:  float
      :value: 0.0



.. py:class:: PairParameters

   .. py:attribute:: donor_type
      :type:  str


   .. py:attribute:: acceptor_type
      :type:  str


   .. py:attribute:: AHdist
      :type:  str


   .. py:attribute:: cosBAH
      :type:  str


   .. py:attribute:: cosAHD
      :type:  str


.. py:class:: HBondDatabaseRaw

   .. py:attribute:: global_parameters
      :type:  GlobalParams


   .. py:attribute:: donor_atom_types
      :type:  Tuple[DonorAtomType, Ellipsis]


   .. py:attribute:: donor_type_params
      :type:  Tuple[DonorTypeParam, Ellipsis]


   .. py:attribute:: acceptor_atom_types
      :type:  Tuple[AcceptorAtomType, Ellipsis]


   .. py:attribute:: acceptor_type_params
      :type:  Tuple[AcceptorTypeParam, Ellipsis]


   .. py:attribute:: pair_parameters
      :type:  Tuple[PairParameters, Ellipsis]


   .. py:attribute:: polynomial_parameters
      :type:  Tuple[PolynomialParameters, Ellipsis]


   .. py:method:: from_file(path)
      :classmethod:



.. py:class:: HBondDatabase

   .. py:attribute:: global_parameters
      :type:  GlobalParams


   .. py:attribute:: donor_atom_types
      :type:  Tuple[DonorAtomType, Ellipsis]


   .. py:attribute:: donor_type_params
      :type:  Tuple[DonorTypeParam, Ellipsis]


   .. py:attribute:: donor_type_mapper
      :type:  pandas.DataFrame


   .. py:attribute:: acceptor_atom_types
      :type:  Tuple[AcceptorAtomType, Ellipsis]


   .. py:attribute:: acceptor_type_params
      :type:  Tuple[AcceptorTypeParam, Ellipsis]


   .. py:attribute:: acceptor_type_mapper
      :type:  pandas.DataFrame


   .. py:attribute:: pair_parameters
      :type:  Tuple[PairParameters, Ellipsis]


   .. py:attribute:: polynomial_parameters
      :type:  Tuple[PolynomialParameters, Ellipsis]


   .. py:method:: from_raw_hbond_db(hbr)
      :classmethod:



   .. py:method:: from_file(path)
      :classmethod:



