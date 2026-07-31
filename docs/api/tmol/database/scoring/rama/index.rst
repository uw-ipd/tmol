tmol.database.scoring.rama
==========================

.. py:module:: tmol.database.scoring.rama


Classes
-------

.. autoapisummary::

   tmol.database.scoring.rama.RamaMappingParams
   tmol.database.scoring.rama.RamaTables
   tmol.database.scoring.rama.RamaDatabase


Module Contents
---------------

.. py:class:: RamaMappingParams

   .. py:attribute:: table_id
      :type:  str


   .. py:attribute:: res_middle
      :type:  str


   .. py:attribute:: res_upper
      :type:  str
      :value: '_'



   .. py:attribute:: invert_phi
      :type:  bool
      :value: False



   .. py:attribute:: invert_psi
      :type:  bool
      :value: False



.. py:class:: RamaTables

   .. py:attribute:: table_id
      :type:  str


   .. py:attribute:: table
      :type:  tmol.types.torch.Tensor[torch.float32]


   .. py:attribute:: bbstep
      :type:  Tuple[float, float]


   .. py:attribute:: bbstart
      :type:  Tuple[float, float]


.. py:class:: RamaDatabase

   .. py:attribute:: uniq_id
      :type:  str


   .. py:attribute:: rama_lookup
      :type:  Tuple[RamaMappingParams, Ellipsis]


   .. py:attribute:: rama_tables
      :type:  Tuple[RamaTables, Ellipsis]


   .. py:method:: from_file(fname: str)
      :classmethod:



