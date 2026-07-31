tmol.pose.packed_block_types
============================

.. py:module:: tmol.pose.packed_block_types


Classes
-------

.. autoapisummary::

   tmol.pose.packed_block_types.PackedBlockTypes


Functions
---------

.. autoapisummary::

   tmol.pose.packed_block_types.residue_types_from_residues


Module Contents
---------------

.. py:function:: residue_types_from_residues(residues)

.. py:class:: PackedBlockTypes

   .. rubric:: Docstring

   .. code-block:: text

      A class to aggregate the properties for a collection of residue types.
      
      The PackedBlockTypes object holds an ordered set of residue types
      (specifically, RefinedResidueTypes); once constructed, this order will not
      change, so residue types may be referred to by index within this object.
      
      The PackedBlockTypes object is the bag in which scoring terms cache their
      tensors holding the chemical/scoring properties of the block types in use.
      Each term needs several tensors in order to map from block-type index to
      the data required to score that block type, and the construction of these
      tensors and moving these tensors can be slowThe idiom we follow to ensure
      that these tensors are preserved between score evaluations is to cache
      them in this object. The term will annotate the PackedBlockTypes object,
      pbt, using setattr(pbt, "tensor_name", tensor) and then will later decide
      if the annotation has already been made using hasattr(pbt, "tensor_name").
      Thus, it is more efficient to use a single PackedBlockTypes object between
      multiple PoseStack objects so that the expense of creating the annotations
      can be amortized of many score evaluations.
      
      Annotation process:
      There are three steps to the annotation process. 1) Terms
      annotate individual block types, 2) terms aggregate (concattenate)
      annotations of the individual block types for the packed_block_type,
      and 3) terms retrieve their annotations. 1) Typically, score terms will
      create one annotation for each of the RefinedResidueType objects that the
      PackedBlockTypes object holds in their method named "setup_block_type,"
      and cache these annotations on each RefinedResiduetype. The idiom we use
      is for these residue-type annotations to be held in numpy arrays (on the
      CPU). 2) The terms will then aggregate the annotations for each of the
      individual residue types into a single torch Tensor, one tensor for each
      property (or property group) the term needs, in their method named
      "setup_packed_block_types"; the idiom is for these annotations to be moved
      to the PackedBlockType's device in this step. 3) Finally, each term will
      retrieve the cached annotations from the PackedBlockType in their
      "render_whole_pose_scoring_module" method.
      

   .. py:attribute:: chem_db
      :type:  tmol.chemical.patched_chemdb.PatchedChemicalDatabase


   .. py:attribute:: restype_set
      :type:  tmol.chemical.restypes.ResidueTypeSet


   .. py:attribute:: active_block_types
      :type:  Sequence[tmol.chemical.restypes.RefinedResidueType]


   .. py:attribute:: restype_index
      :type:  pandas.Index


   .. py:attribute:: max_n_atoms
      :type:  int


   .. py:attribute:: n_atoms
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: atom_is_real
      :type:  tmol.types.torch.Tensor[torch.uint8][:, :]


   .. py:attribute:: atom_is_hydrogen
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: atom_downstream_of_conn
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, :]


   .. py:attribute:: atom_paths_from_conn
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, tmol.chemical.constants.MAX_PATHS_FROM_CONNECTION, 3]


   .. py:attribute:: max_n_torsions
      :type:  int


   .. py:attribute:: n_torsions
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: torsion_is_real
      :type:  tmol.types.torch.Tensor[torch.uint8][:, :]


   .. py:attribute:: torsion_uaids
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, 3]


   .. py:attribute:: is_torsion_mc
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: n_mc_torsions
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: mc_torsion_is_real
      :type:  tmol.types.torch.Tensor[torch.uint8][:, :]


   .. py:attribute:: mc_torsions
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: n_sc_torsions
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: sc_torsion_is_real
      :type:  tmol.types.torch.Tensor[torch.uint8][:, :]


   .. py:attribute:: sc_torsions
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: which_mcsc_torsions
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: max_n_bonds
      :type:  int


   .. py:attribute:: n_bonds
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: bond_is_real
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: bond_indices
      :type:  tmol.types.torch.Tensor[torch.int32][:, :, 2]


   .. py:attribute:: max_n_conn
      :type:  int


   .. py:attribute:: n_conn
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: conn_is_real
      :type:  tmol.types.torch.Tensor[torch.bool][:, :]


   .. py:attribute:: conn_atom
      :type:  tmol.types.torch.Tensor[torch.int32][:, :]


   .. py:attribute:: down_conn_inds
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: up_conn_inds
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: polymeric_conn_inds
      :type:  tmol.types.torch.Tensor[torch.int32][:, 2]


   .. py:attribute:: default_jump_connection_atom_inds
      :type:  tmol.types.torch.Tensor[torch.int32][:]


   .. py:attribute:: device
      :type:  torch.device


   .. py:property:: n_types


   .. py:method:: from_restype_list(chem_db: tmol.chemical.patched_chemdb.PatchedChemicalDatabase, restype_set: tmol.chemical.restypes.ResidueTypeSet, active_block_types: Sequence[tmol.chemical.restypes.RefinedResidueType], device: torch.device)
      :classmethod:



   .. py:method:: count_max_n_atoms(active_block_types: Sequence[tmol.chemical.restypes.RefinedResidueType])
      :classmethod:



   .. py:method:: count_n_atoms(active_block_types: Sequence[tmol.chemical.restypes.RefinedResidueType], device: torch.device)
      :classmethod:



   .. py:method:: determine_real_atoms(max_n_atoms: int, n_atoms: tmol.types.torch.Tensor[torch.int32][:], device: torch.device)
      :classmethod:



   .. py:method:: determine_h_atoms(chem_db: tmol.chemical.patched_chemdb.PatchedChemicalDatabase, max_n_atoms: int, n_atoms: tmol.types.torch.Tensor[torch.int32][:], active_block_types, device: torch.device)
      :classmethod:



   .. py:method:: join_atom_downstream_of_conn(active_block_types: Sequence[tmol.chemical.restypes.RefinedResidueType], device: torch.device)
      :classmethod:



   .. py:method:: join_atom_paths_from_conn(active_block_types: Sequence[tmol.chemical.restypes.RefinedResidueType], device: torch.device)
      :classmethod:



   .. py:method:: join_torsion_uaids(active_block_types, device)
      :classmethod:



   .. py:method:: join_is_torsion_mcs(active_block_types, device)
      :classmethod:



   .. py:method:: join_mc_torsion_inds(active_block_types, device)
      :classmethod:



   .. py:method:: join_sc_torsion_inds(active_block_types, device)
      :classmethod:



   .. py:method:: join_mcsc_torsion_inds(active_block_types, device)
      :classmethod:



   .. py:method:: join_bond_indices(active_block_types, device)
      :classmethod:



   .. py:method:: join_conn_indices(active_block_types, device)
      :classmethod:



   .. py:method:: join_polymeric_connections(active_block_types, device)
      :classmethod:



   .. py:method:: join_default_jump_connection_atom_inds(active_block_types, device)
      :classmethod:



   .. py:method:: inds_for_restypes(res_types: Sequence[tmol.chemical.restypes.RefinedResidueType])


   .. py:method:: cpu()


