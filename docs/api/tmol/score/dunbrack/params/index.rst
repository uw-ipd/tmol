tmol.score.dunbrack.params
==========================

.. py:module:: tmol.score.dunbrack.params


Classes
-------

.. autoapisummary::

   tmol.score.dunbrack.params.DunbrackParams
   tmol.score.dunbrack.params.DunbrackScratch
   tmol.score.dunbrack.params.ScoringDunbrackDatabaseView
   tmol.score.dunbrack.params.ScoringDunbrackDatabaseAux
   tmol.score.dunbrack.params.SamplingDunbrackDatabaseView
   tmol.score.dunbrack.params.DunbrackParamResolver


Module Contents
---------------

.. py:class:: DunbrackParams

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`


   .. py:attribute:: ndihe_for_res
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: dihedral_offset_for_res
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: dihedral_atom_inds
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, 4]


   .. py:attribute:: rottable_set_for_res
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: nchi_for_res
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: nrotameric_chi_for_res
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: rotres2resid
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: prob_table_offset_for_rotresidue
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: rotmean_table_offset_for_residue
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: rotind2tableind_offset_for_res
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: rotameric_chi_desc
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, 2]


   .. py:attribute:: semirotameric_chi_desc
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, 4]


.. py:class:: DunbrackScratch

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`


   .. py:attribute:: dihedrals
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: ddihe_dxyz
      :type:  tmol.types.torch.Tensor[torch.float][:, :, 4, 3]


   .. py:attribute:: rotameric_rottable_assignment
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: semirotameric_rottable_assignment
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


.. py:class:: ScoringDunbrackDatabaseView

   Bases: :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      The tables for the dunbrack database needed for scoring
      stored on the device
      

   .. py:attribute:: rotameric_neglnprob_tables
      :type:  tmol.types.torch.Tensor[torch.float][:, :, :]


   .. py:attribute:: rotprob_table_sizes
      :type:  tmol.types.torch.Tensor[torch.long][:, 2]


   .. py:attribute:: rotprob_table_strides
      :type:  tmol.types.torch.Tensor[torch.long][:, 2]


   .. py:attribute:: rotameric_mean_tables
      :type:  tmol.types.torch.Tensor[torch.float][:, :, :]


   .. py:attribute:: rotameric_sdev_tables
      :type:  tmol.types.torch.Tensor[torch.float][:, :, :]


   .. py:attribute:: rotmean_table_sizes
      :type:  tmol.types.torch.Tensor[torch.long][:, 2]


   .. py:attribute:: rotmean_table_strides
      :type:  tmol.types.torch.Tensor[torch.long][:, 2]


   .. py:attribute:: rotameric_bb_start
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: rotameric_bb_step
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: rotameric_bb_periodicity
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: rotameric_rotind2tableind
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: semirotameric_rotind2tableind
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: semirotameric_tables
      :type:  tmol.types.torch.Tensor[torch.float][:, :, :, :]


   .. py:attribute:: semirot_table_sizes
      :type:  tmol.types.torch.Tensor[torch.long][:, 3]


   .. py:attribute:: semirot_table_strides
      :type:  tmol.types.torch.Tensor[torch.long][:, 3]


   .. py:attribute:: semirot_start
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: semirot_step
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: semirot_periodicity
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


.. py:class:: ScoringDunbrackDatabaseAux

   Bases: :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. py:attribute:: rotameric_prob_tableset_offsets
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: rotameric_meansdev_tableset_offsets
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: nchi_for_table_set
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: rotameric_chi_ri2ti_offsets
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: semirotameric_tableset_offsets
      :type:  tmol.types.torch.Tensor[torch.int32][:]


.. py:class:: SamplingDunbrackDatabaseView

   Bases: :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      The tables that are needed in order to sample
      side-chain conformations.
      

   .. py:attribute:: rotameric_prob_tables
      :type:  tmol.types.torch.Tensor[torch.float][:, :, :]


   .. py:attribute:: rotprob_table_sizes
      :type:  tmol.types.torch.Tensor[torch.long][:, 2]


   .. py:attribute:: rotprob_table_strides
      :type:  tmol.types.torch.Tensor[torch.long][:, 2]


   .. py:attribute:: rotameric_mean_tables
      :type:  tmol.types.torch.Tensor[torch.float][:, :, :]


   .. py:attribute:: rotameric_sdev_tables
      :type:  tmol.types.torch.Tensor[torch.float][:, :, :]


   .. py:attribute:: rotmean_table_sizes
      :type:  tmol.types.torch.Tensor[torch.long][:, 2]


   .. py:attribute:: rotmean_table_strides
      :type:  tmol.types.torch.Tensor[torch.long][:, 2]


   .. py:attribute:: rotameric_meansdev_tableset_offsets
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: n_rotamers_for_tableset
      :type:  tmol.types.torch.Tensor[torch.long][:]


   .. py:attribute:: n_rotamers_for_tableset_offsets
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: sorted_rotamer_2_rotamer
      :type:  tmol.types.torch.Tensor[torch.long][:, :, :]


   .. py:attribute:: rotameric_bb_start
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: rotameric_bb_step
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: rotameric_bb_periodicity
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: rotameric_rotind2tableind
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: semirotameric_rotind2tableind
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: all_chi_rotind2tableind
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: all_chi_rotind2tableind_offsets
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: semirotameric_tables
      :type:  tmol.types.torch.Tensor[torch.float][:, :, :, :]


   .. py:attribute:: semirot_table_sizes
      :type:  tmol.types.torch.Tensor[torch.long][:, 3]


   .. py:attribute:: semirot_table_strides
      :type:  tmol.types.torch.Tensor[torch.long][:, 3]


   .. py:attribute:: semirot_start
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: semirot_step
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: semirot_periodicity
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: nchi_for_table_set
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: rotwells
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


.. py:class:: DunbrackParamResolver

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: scoring_db
      :type:  ScoringDunbrackDatabaseView


   .. py:attribute:: scoring_db_aux
      :type:  ScoringDunbrackDatabaseAux


   .. py:attribute:: sampling_db
      :type:  SamplingDunbrackDatabaseView


   .. py:attribute:: all_table_indices
      :type:  pandas.DataFrame


   .. py:attribute:: rotameric_table_indices
      :type:  pandas.DataFrame


   .. py:attribute:: semirotameric_table_indices
      :type:  pandas.DataFrame


   .. py:attribute:: device
      :type:  torch.device


   .. py:method:: from_database(dun_database: tmol.database.scoring.dunbrack_libraries.DunbrackRotamerLibrary, device: torch.device)
      :classmethod:



   .. py:method:: create_sorted_rot_2_rot(all_rotlibs, device)
      :classmethod:



   .. py:method:: create_rotamer_well_table(all_rotlibs, device)
      :classmethod:



