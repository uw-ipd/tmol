tmol.score.backbone_torsion.params
==================================

.. py:module:: tmol.score.backbone_torsion.params


Classes
-------

.. autoapisummary::

   tmol.score.backbone_torsion.params.PackedRamaDatabase
   tmol.score.backbone_torsion.params.PackedOmegaDatabase
   tmol.score.backbone_torsion.params.BackboneTorsionParamResolver


Module Contents
---------------

.. py:class:: PackedRamaDatabase

   Bases: :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. py:attribute:: tables
      :type:  tmol.types.torch.Tensor[torch.float][:, :, :]


   .. py:attribute:: bbsteps
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: bbstarts
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


.. py:class:: PackedOmegaDatabase

   Bases: :py:obj:`tmol.types.attrs.ConvertAttrs`


   .. py:attribute:: tables
      :type:  tmol.types.torch.Tensor[torch.float][:, 2, :, :]


   .. py:attribute:: bbsteps
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


   .. py:attribute:: bbstarts
      :type:  tmol.types.torch.Tensor[torch.float][:, :]


.. py:class:: BackboneTorsionParamResolver

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: rama_lookup
      :type:  pandas.DataFrame


   .. py:attribute:: omega_lookup
      :type:  pandas.DataFrame


   .. py:attribute:: rama_params
      :type:  PackedRamaDatabase


   .. py:attribute:: omega_params
      :type:  PackedOmegaDatabase


   .. py:attribute:: device
      :type:  torch.device


   .. py:method:: from_database(rama_database: tmol.database.scoring.rama.RamaDatabase, bbdep_omega_database: tmol.database.scoring.omega_bbdep.OmegaBBDepDatabase, device: torch.device)
      :classmethod:



