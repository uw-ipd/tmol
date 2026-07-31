tmol.score.elec.params
======================

.. py:module:: tmol.score.elec.params


Classes
-------

.. autoapisummary::

   tmol.score.elec.params.ElecGlobalParams
   tmol.score.elec.params.ElecParamResolver


Module Contents
---------------

.. py:class:: ElecGlobalParams

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: elec_min_dis
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: elec_max_dis
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: elec_sigmoidal_die_D
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: elec_sigmoidal_die_D0
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: elec_sigmoidal_die_S
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


.. py:class:: ElecParamResolver

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      Container for global/type/pair parameters, indexed by atom type name.
      
      Param resolver stores pair parameters for a collection of atom types, using
      a pandas Index to map from string atom type to a resolver-specific integer type
      index.
      

   .. py:attribute:: global_params
      :type:  ElecGlobalParams


   .. py:attribute:: device
      :type:  torch.device


   .. py:attribute:: cp_reps
      :type:  dict


   .. py:attribute:: partial_charges
      :type:  dict


   .. py:method:: get_partial_charges_for_block(block_type: tmol.chemical.restypes.RefinedResidueType)


   .. py:method:: get_bonded_path_length_mapping_for_block(block_type: tmol.chemical.restypes.RefinedResidueType)

      .. rubric:: Docstring

      .. code-block:: text

         remap bonded path length for a residue block
         


   .. py:method:: from_database(elec_database: tmol.database.scoring.elec.ElecDatabase, device: torch.device)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Initialize param resolver for all atoms defined in database.
         


