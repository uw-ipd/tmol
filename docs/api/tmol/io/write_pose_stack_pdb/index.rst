tmol.io.write_pose_stack_pdb
============================

.. py:module:: tmol.io.write_pose_stack_pdb


Functions
---------

.. autoapisummary::

   tmol.io.write_pose_stack_pdb.write_pose_stack_pdb
   tmol.io.write_pose_stack_pdb.atom_records_from_pose_stack
   tmol.io.write_pose_stack_pdb.atom_records_from_coords


Module Contents
---------------

.. py:function:: write_pose_stack_pdb(pose_stack: tmol.pose.pose_stack.PoseStack, fname_out: str, merge_fragments: bool = True, **kwargs)

   .. rubric:: Docstring

   .. code-block:: text

      Write a PDB-formatted file to disk given an input PoseStack.
      Optionally, additional arguments may be passed to the inner function
      "atom_records_from_pose_stack." Fragmented ligands use their original
      residue identity by default; pass ``merge_fragments=False`` to keep
      fragment residues separate.
      

.. py:function:: atom_records_from_pose_stack(pose_stack: tmol.pose.pose_stack.PoseStack, merge_fragments: bool = True) -> tmol.types.array.NDArray[tmol.io.pdb_parsing.atom_record_dtype][:]

   .. rubric:: Docstring

   .. code-block:: text

      Create a numpy array holding the atom records needed to write a
      PDB file from a PoseStack.
      
      Fragmented ligands use their original residue identity by default. Pass
      ``merge_fragments=False`` to retain the separate fragment residue numbers.
      

.. py:function:: atom_records_from_coords(pbt: tmol.pose.packed_block_types.PackedBlockTypes, chain_ind_for_block: tmol.types.torch.Tensor[torch.int64][:, :], block_types64: tmol.types.torch.Tensor[torch.int64][:, :], pose_like_coords: tmol.types.torch.Tensor[torch.float32][:, :, 3], block_coord_offset: tmol.types.torch.Tensor[torch.int32][:, :], residue_labels: Optional[tmol.types.array.NDArray[int][:, :]], residue_insertion_codes: Optional[tmol.types.array.NDArray[object][:, :]], chain_labels: Optional[tmol.types.array.NDArray[object][:, :]], atom_occupancy: Optional[tmol.types.array.NDArray[numpy.float32][:, :]], atom_b_factor: Optional[tmol.types.array.NDArray[numpy.float32][:, :]]) -> tmol.types.array.NDArray[tmol.io.pdb_parsing.atom_record_dtype][:]

   .. rubric:: Docstring

   .. code-block:: text

      Create a numpy array holding the atom records needed to write a
      PDB file from the coordinates and block types of a stack of structures,
      laid out in pose-stack form.
      

