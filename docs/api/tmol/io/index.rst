tmol.io
=======

.. py:module:: tmol.io


Submodules
----------

.. toctree::
   :maxdepth: 1

   /api/tmol/io/canonical_form/index
   /api/tmol/io/canonical_ordering/index
   /api/tmol/io/chain_deduction/index
   /api/tmol/io/extern/index
   /api/tmol/io/generic/index
   /api/tmol/io/pdb_parsing/index
   /api/tmol/io/pose_stack_construction/index
   /api/tmol/io/pose_stack_deconstruction/index
   /api/tmol/io/pose_stack_from_atomworks/index
   /api/tmol/io/pose_stack_from_biotite/index
   /api/tmol/io/pose_stack_from_openfold/index
   /api/tmol/io/pose_stack_from_rosettafold2/index
   /api/tmol/io/write_pose_stack_pdb/index


Functions
---------

.. autoapisummary::

   tmol.io.pose_stack_from_pdb


Package Contents
----------------

.. py:function:: pose_stack_from_pdb(pdb_lines_or_fname: Union[str, list], device: torch.device, *, residue_start: Optional[int] = None, residue_end: Optional[int] = None, res_not_connected: Optional[tmol.types.torch.Tensor[torch.bool][:, :, 2]] = None, **kwargs) -> tmol.pose.pose_stack.PoseStack

   .. rubric:: Docstring

   .. code-block:: text

      Construct a PoseStack given the contents of a PDB file or the name of a PDB file,
      using the full set of residue types contained in tmol's chemical.yaml file.
      
      Optionally, a subset of the residues in the range from residue_start to residue_end-1
      can be requested.
      Any additional keyword arguments will be passed to pose_stack_from_canonical_form
      

