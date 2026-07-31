tmol.database.scoring.dunbrack_libraries
========================================

.. py:module:: tmol.database.scoring.dunbrack_libraries


Classes
-------

.. autoapisummary::

   tmol.database.scoring.dunbrack_libraries.RotamericDataForAA
   tmol.database.scoring.dunbrack_libraries.RotamericAADunbrackLibrary
   tmol.database.scoring.dunbrack_libraries.SemiRotamericAADunbrackLibrary
   tmol.database.scoring.dunbrack_libraries.DunMappingParams
   tmol.database.scoring.dunbrack_libraries.DunbrackRotamerLibrary


Module Contents
---------------

.. py:class:: RotamericDataForAA

   .. py:attribute:: rotamers
      :type:  tmol.types.torch.Tensor[int][:, :]


   .. py:attribute:: rotamer_probabilities
      :type:  tmol.types.torch.Tensor[float]


   .. py:attribute:: rotamer_means
      :type:  tmol.types.torch.Tensor[float]


   .. py:attribute:: rotamer_stdvs
      :type:  tmol.types.torch.Tensor[float]


   .. py:attribute:: prob_sorted_rot_inds
      :type:  tmol.types.torch.Tensor[int]


   .. py:attribute:: backbone_dihedral_start
      :type:  tmol.types.torch.Tensor[float][:]


   .. py:attribute:: backbone_dihedral_step
      :type:  tmol.types.torch.Tensor[float][:]


   .. py:attribute:: rotamer_alias
      :type:  tmol.types.torch.Tensor[int][:, :]


   .. py:method:: nrotamers()


   .. py:method:: nchi()


.. py:class:: RotamericAADunbrackLibrary

   .. py:attribute:: table_name
      :type:  str


   .. py:attribute:: rotameric_data
      :type:  RotamericDataForAA


.. py:class:: SemiRotamericAADunbrackLibrary

   .. py:attribute:: table_name
      :type:  str


   .. py:attribute:: rotameric_data
      :type:  RotamericDataForAA


   .. py:attribute:: non_rot_chi_start
      :type:  float


   .. py:attribute:: non_rot_chi_step
      :type:  float


   .. py:attribute:: non_rot_chi_period
      :type:  float


   .. py:attribute:: rotameric_chi_rotamers
      :type:  tmol.types.torch.Tensor[int][:, :]


   .. py:attribute:: nonrotameric_chi_probabilities
      :type:  tmol.types.torch.Tensor[float]


   .. py:attribute:: rotamer_boundaries
      :type:  tmol.types.torch.Tensor[float][:, 2]


.. py:class:: DunMappingParams

   .. py:attribute:: dun_table_name
      :type:  str


   .. py:attribute:: residue_name
      :type:  str


.. py:class:: DunbrackRotamerLibrary

   .. py:attribute:: dun_lookup
      :type:  Tuple[DunMappingParams, Ellipsis]


   .. py:attribute:: rotameric_libraries
      :type:  Tuple[RotamericAADunbrackLibrary, Ellipsis]


   .. py:attribute:: semi_rotameric_libraries
      :type:  Tuple[SemiRotamericAADunbrackLibrary, Ellipsis]


   .. py:method:: from_file(fname: str)
      :classmethod:



