tmol.pack.rotamer.mainchain_fingerprint
=======================================

.. py:module:: tmol.pack.rotamer.mainchain_fingerprint


Classes
-------

.. autoapisummary::

   tmol.pack.rotamer.mainchain_fingerprint.AtomFingerprint
   tmol.pack.rotamer.mainchain_fingerprint.MCFingerprint
   tmol.pack.rotamer.mainchain_fingerprint.MCFingerprints


Functions
---------

.. autoapisummary::

   tmol.pack.rotamer.mainchain_fingerprint.create_non_sidechain_fingerprint
   tmol.pack.rotamer.mainchain_fingerprint.create_mainchain_fingerprint
   tmol.pack.rotamer.mainchain_fingerprint.annotate_residue_type_with_sampler_fingerprints
   tmol.pack.rotamer.mainchain_fingerprint.find_max_length_fp_among_res_samplers
   tmol.pack.rotamer.mainchain_fingerprint.find_unique_fingerprints


Module Contents
---------------

.. py:class:: AtomFingerprint

   .. py:attribute:: mc_ind
      :type:  int


   .. py:attribute:: mc_bond_dist
      :type:  int


   .. py:attribute:: chirality
      :type:  int


   .. py:attribute:: element
      :type:  int


   .. py:attribute:: duplicate_index
      :type:  int
      :value: 0



.. py:class:: MCFingerprint

   .. py:attribute:: mc_ats
      :type:  tmol.types.array.NDArray[numpy.int32][:]


   .. py:attribute:: mc_at_fingerprints
      :type:  Tuple[AtomFingerprint, Ellipsis]


   .. py:attribute:: fingerprint
      :type:  Tuple[AtomFingerprint, Ellipsis]


   .. py:attribute:: at_for_fingerprint
      :type:  Mapping[AtomFingerprint, int]


.. py:class:: MCFingerprints

   .. py:attribute:: atom_mapping
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :, :]


   .. py:attribute:: sampler_mapping
      :type:  Mapping[str, int]


   .. py:attribute:: max_sampler
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: max_fingerprint
      :type:  tmol.types.torch.Tensor[torch.int32][:]


.. py:function:: create_non_sidechain_fingerprint(rt: tmol.chemical.restypes.RefinedResidueType, parents: tmol.types.array.NDArray[numpy.int32][:], sc_atoms: tmol.types.array.NDArray[numpy.int32][:], chem_db: tmol.chemical.patched_chemdb.PatchedChemicalDatabase)

.. py:function:: create_mainchain_fingerprint(rt: tmol.chemical.restypes.RefinedResidueType, sc_roots: Tuple[str, Ellipsis], chem_db: tmol.chemical.patched_chemdb.PatchedChemicalDatabase)

.. py:function:: annotate_residue_type_with_sampler_fingerprints(restype: tmol.chemical.restypes.RefinedResidueType, samplers: Tuple[tmol.pack.rotamer.chi_sampler.ChiSampler, Ellipsis], chem_db: tmol.chemical.patched_chemdb.PatchedChemicalDatabase)

.. py:function:: find_max_length_fp_among_res_samplers(pbt: tmol.pose.packed_block_types.PackedBlockTypes, sampler_types, fp_sets)

.. py:function:: find_unique_fingerprints(pbt: tmol.pose.packed_block_types.PackedBlockTypes)

