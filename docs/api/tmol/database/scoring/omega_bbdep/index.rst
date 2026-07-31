tmol.database.scoring.omega_bbdep
=================================

.. py:module:: tmol.database.scoring.omega_bbdep


Classes
-------

.. autoapisummary::

   tmol.database.scoring.omega_bbdep.OmegaBBDepMappingParams
   tmol.database.scoring.omega_bbdep.OmegaBBDepTables
   tmol.database.scoring.omega_bbdep.OmegaBBDepDatabase


Module Contents
---------------

.. py:class:: OmegaBBDepMappingParams

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



.. py:class:: OmegaBBDepTables

   .. py:attribute:: table_id
      :type:  str


   .. py:attribute:: mu
      :type:  tmol.types.torch.Tensor[torch.float32]


   .. py:attribute:: sigma
      :type:  tmol.types.torch.Tensor[torch.float32]


   .. py:attribute:: bbstep
      :type:  Tuple[float, float]


   .. py:attribute:: bbstart
      :type:  Tuple[float, float]


.. py:class:: OmegaBBDepDatabase

   .. py:attribute:: uniq_id
      :type:  str


   .. py:attribute:: bbdep_omega_lookup
      :type:  Tuple[OmegaBBDepMappingParams, Ellipsis]


   .. py:attribute:: bbdep_omega_tables
      :type:  Tuple[OmegaBBDepTables, Ellipsis]


   .. py:method:: from_file(fname: str)
      :classmethod:



