tmol.score.hbond.params
=======================

.. py:module:: tmol.score.hbond.params


Classes
-------

.. autoapisummary::

   tmol.score.hbond.params.HBondPolyParams
   tmol.score.hbond.params.HBondPairParams
   tmol.score.hbond.params.HBondParamResolver
   tmol.score.hbond.params.CompactedHBondDatabase


Module Contents
---------------

.. py:class:: HBondPolyParams

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. py:attribute:: range
      :type:  tmol.types.torch.Tensor[torch.double][Ellipsis, 2]


   .. py:attribute:: bound
      :type:  tmol.types.torch.Tensor[torch.double][Ellipsis, 2]


   .. py:attribute:: coeffs
      :type:  tmol.types.torch.Tensor[torch.double][Ellipsis, 11]


   .. py:method:: to(device: torch.device)

      .. rubric:: Docstring

      .. code-block:: text

         Perform dtype/device conversion for all subtensors.
         
         Note that this may be an invalid operations if the TensorGroup contains
         heterogenous tensor dtypes.
         
         Performs Tensor dtype and/or device conversion. A :class:`torch.dtype`
         and :class:`torch.device` are inferred from the arguments of
         ``self.to(*args, **kwargs)``.
         
         If all subtensors already have the correct dtype and device then
         ``self`` is returned.
         


   .. py:method:: full(shape, fill_value)
      :classmethod:



.. py:class:: HBondPairParams

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: donor_weight
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: acceptor_weight
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: acceptor_hybridization
      :type:  tmol.types.torch.Tensor[torch.int32][Ellipsis]


   .. py:attribute:: AHdist
      :type:  HBondPolyParams


   .. py:attribute:: cosBAH
      :type:  HBondPolyParams


   .. py:attribute:: cosAHD
      :type:  HBondPolyParams


   .. py:method:: to(device: torch.device)

      .. rubric:: Docstring

      .. code-block:: text

         Perform dtype/device conversion for all subtensors.
         
         Note that this may be an invalid operations if the TensorGroup contains
         heterogenous tensor dtypes.
         
         Performs Tensor dtype and/or device conversion. A :class:`torch.dtype`
         and :class:`torch.device` are inferred from the arguments of
         ``self.to(*args, **kwargs)``.
         
         If all subtensors already have the correct dtype and device then
         ``self`` is returned.
         


   .. py:method:: full(shape, fill_value)
      :classmethod:



.. py:class:: HBondParamResolver

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: donor_type_index
      :type:  pandas.Index


   .. py:attribute:: acceptor_type_index
      :type:  pandas.Index


   .. py:attribute:: pair_params
      :type:  HBondPairParams


   .. py:attribute:: device
      :type:  torch.device


   .. py:method:: from_database(chemical_database: tmol.database.chemical.ChemicalDatabase, hbond_database: tmol.database.scoring.hbond.HBondDatabase, device: torch.device)
      :classmethod:



.. py:class:: CompactedHBondDatabase

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      Store the hbond evaluation parameters in a compact form
      

   .. py:attribute:: global_param_table
      :type:  tmol.types.torch.Tensor[torch.float32][:, :]


   .. py:attribute:: pair_param_table
      :type:  tmol.types.torch.Tensor[torch.float32][:, :, :]


   .. py:attribute:: pair_poly_table
      :type:  tmol.types.torch.Tensor[torch.float64][:, :, :]


   .. py:method:: from_database(chemical_database: tmol.database.chemical.ChemicalDatabase, hbond_database: tmol.database.scoring.hbond.HBondDatabase, device: torch.device, /)
      :classmethod:



