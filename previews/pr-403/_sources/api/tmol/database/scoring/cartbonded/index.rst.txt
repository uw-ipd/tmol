tmol.database.scoring.cartbonded
================================

.. py:module:: tmol.database.scoring.cartbonded


Classes
-------

.. autoapisummary::

   tmol.database.scoring.cartbonded.LengthGroup
   tmol.database.scoring.cartbonded.AngleGroup
   tmol.database.scoring.cartbonded.TorsionGroup
   tmol.database.scoring.cartbonded.ImproperGroup
   tmol.database.scoring.cartbonded.HxlTorsionGroup
   tmol.database.scoring.cartbonded.CartRes
   tmol.database.scoring.cartbonded.CartBondedDatabase


Module Contents
---------------

.. py:class:: LengthGroup

   .. py:attribute:: atm1
      :type:  str


   .. py:attribute:: atm2
      :type:  str


   .. py:attribute:: x0
      :type:  float


   .. py:attribute:: K
      :type:  float


   .. py:attribute:: type
      :type:  int
      :value: 0



.. py:class:: AngleGroup

   .. py:attribute:: atm1
      :type:  str


   .. py:attribute:: atm2
      :type:  str


   .. py:attribute:: atm3
      :type:  str


   .. py:attribute:: x0
      :type:  float


   .. py:attribute:: K
      :type:  float


   .. py:attribute:: type
      :type:  int
      :value: 1



.. py:class:: TorsionGroup

   .. py:attribute:: atm1
      :type:  str


   .. py:attribute:: atm2
      :type:  str


   .. py:attribute:: atm3
      :type:  str


   .. py:attribute:: atm4
      :type:  str


   .. py:attribute:: k1
      :type:  float
      :value: 0.0



   .. py:attribute:: phi1
      :type:  float
      :value: 0.0



   .. py:attribute:: k2
      :type:  float
      :value: 0.0



   .. py:attribute:: phi2
      :type:  float
      :value: 0.0



   .. py:attribute:: k3
      :type:  float
      :value: 0.0



   .. py:attribute:: phi3
      :type:  float
      :value: 0.0



   .. py:attribute:: type
      :type:  int
      :value: 2



.. py:class:: ImproperGroup

   Bases: :py:obj:`TorsionGroup`


   .. py:attribute:: type
      :type:  int
      :value: 3



.. py:class:: HxlTorsionGroup

   Bases: :py:obj:`TorsionGroup`


   .. py:attribute:: type
      :type:  int
      :value: 4



.. py:class:: CartRes

   .. py:attribute:: length_parameters
      :type:  Tuple[LengthGroup, Ellipsis]


   .. py:attribute:: angle_parameters
      :type:  Tuple[AngleGroup, Ellipsis]


   .. py:attribute:: torsion_parameters
      :type:  Tuple[TorsionGroup, Ellipsis]


   .. py:attribute:: improper_parameters
      :type:  Tuple[ImproperGroup, Ellipsis]


   .. py:attribute:: hxltorsion_parameters
      :type:  Tuple[HxlTorsionGroup, Ellipsis]


.. py:class:: CartBondedDatabase

   .. py:attribute:: residue_params
      :type:  dict[str, CartRes]


   .. py:attribute:: hash
      :type:  str


   .. py:method:: from_file(path)
      :classmethod:



   .. py:method:: from_cartres_dict(cartres_dict: dict[str, CartRes])
      :classmethod:



