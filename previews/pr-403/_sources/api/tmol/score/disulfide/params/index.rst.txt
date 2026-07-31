tmol.score.disulfide.params
===========================

.. py:module:: tmol.score.disulfide.params


Classes
-------

.. autoapisummary::

   tmol.score.disulfide.params.DisulfideGlobalParams


Module Contents
---------------

.. py:class:: DisulfideGlobalParams

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`


   .. py:attribute:: d_location
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: d_scale
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: d_shape
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: a_logA
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: a_kappa
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: a_mu
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dss_logA1
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dss_kappa1
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dss_mu1
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dss_logA2
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dss_kappa2
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dss_mu2
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dcs_logA1
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dcs_mu1
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dcs_kappa1
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dcs_logA2
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dcs_mu2
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dcs_kappa2
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dcs_logA3
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dcs_mu3
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: dcs_kappa3
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: wt_dih_ss
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: wt_dih_cs
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: wt_ang
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: wt_len
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:attribute:: shift
      :type:  tmol.types.torch.Tensor[torch.float32][Ellipsis]


   .. py:method:: from_database(disulfide_database: tmol.database.scoring.disulfide.DisulfideDatabase, device: torch.device)
      :classmethod:



