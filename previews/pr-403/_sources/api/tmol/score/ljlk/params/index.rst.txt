tmol.score.ljlk.params
======================

.. py:module:: tmol.score.ljlk.params


Classes
-------

.. autoapisummary::

   tmol.score.ljlk.params.LJLKGlobalParams
   tmol.score.ljlk.params.LJLKTypeParams
   tmol.score.ljlk.params.LJLKParamResolver


Module Contents
---------------

.. py:class:: LJLKGlobalParams

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`


   .. py:attribute:: max_dis
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lj_dlin_sigma_factor
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lj_dlin_sigma_factor_soft
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lj_hbond_OH_donor_dis
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lj_hbond_dis
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lj_hbond_hdis
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lk_min_dis2sigma
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lkb_water_dist
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lkb_water_angle_sp2
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lkb_water_angle_sp3
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lkb_water_angle_ring
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lkb_water_tors_sp2
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis, :]


   .. py:attribute:: lkb_water_tors_sp3
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis, :]


   .. py:attribute:: lkb_water_tors_ring
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis, :]


.. py:class:: LJLKTypeParams

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`


   .. py:attribute:: lj_radius
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lj_wdepth
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lk_dgfree
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lk_lambda
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: lk_volume
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: is_acceptor
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


   .. py:attribute:: acceptor_hybridization
      :type:  tmol.types.torch.Tensor[torch.int][Ellipsis]


   .. py:attribute:: is_donor
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


   .. py:attribute:: is_hydroxyl
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


   .. py:attribute:: is_polarh
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


   .. py:attribute:: is_hydrogen
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


   .. py:attribute:: is_carbon_lk
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


.. py:class:: LJLKParamResolver

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      Container for global/type/pair parameters, indexed by atom type name.
      
      Param resolver stores pair parameters for a collection of atom types, using
      a pandas Index to map from string atom type to a resolver-specific integer type
      index.
      

   .. py:attribute:: atom_type_index
      :type:  pandas.Index


   .. py:attribute:: global_params
      :type:  LJLKGlobalParams


   .. py:attribute:: type_params
      :type:  LJLKTypeParams


   .. py:attribute:: device
      :type:  torch.device


   .. py:method:: from_database(chemical_database: tmol.chemical.patched_chemdb.PatchedChemicalDatabase, ljlk_database: tmol.database.scoring.ljlk.LJLKDatabase, device: torch.device)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Initialize param resolver for all atom types in database.
         


   .. py:method:: from_param_resolver(atom_type_resolver: tmol.score.chemical_database.AtomTypeParamResolver, ljlk_database: tmol.database.scoring.ljlk.LJLKDatabase)
      :classmethod:



