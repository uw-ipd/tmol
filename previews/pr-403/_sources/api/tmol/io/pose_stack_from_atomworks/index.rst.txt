tmol.io.pose_stack_from_atomworks
=================================

.. py:module:: tmol.io.pose_stack_from_atomworks


Attributes
----------

.. autoapisummary::

   tmol.io.pose_stack_from_atomworks.ATOMWORKS_NAME3S
   tmol.io.pose_stack_from_atomworks.ATOMWORKS_ATOM37_NAMES


Functions
---------

.. autoapisummary::

   tmol.io.pose_stack_from_atomworks.pose_stack_from_atomworks
   tmol.io.pose_stack_from_atomworks.canonical_form_from_atomworks
   tmol.io.pose_stack_from_atomworks.atomworks_from_pose_stack
   tmol.io.pose_stack_from_atomworks.canonical_ordering_for_atomworks
   tmol.io.pose_stack_from_atomworks.packed_block_types_for_atomworks


Module Contents
---------------

.. py:data:: ATOMWORKS_NAME3S
   :value: ['<M>', 'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS',...


.. py:data:: ATOMWORKS_ATOM37_NAMES

.. py:function:: pose_stack_from_atomworks(coords: torch.Tensor, residue_type: torch.Tensor, chain_iid: torch.Tensor, **kwargs) -> tmol.pose.pose_stack.PoseStack

   .. rubric:: Docstring

   .. code-block:: text

      Build a PoseStack from atomworks UNIFIED_ATOM37_ENCODING tensors.
      
      This function will build a PoseStack using a limited set of residue types:
      only the canonical amino acids with the canonical n- and c-termini patches.
      It begins by constructing a "canonical form" and then passes that canonical
      form to the pose_stack_from_canonical_form function.
      
      :param coords: Atom coordinates in the atomworks atom37 layout.
      :type coords: Tensor, shape [batch, n_res, 37, 3]
      :param residue_type: Atomworks token indices. Must be in 1..20 (standard protein only).
      :type residue_type: Tensor[int64], shape [batch, n_res]
      :param chain_iid: Chain identifiers (integer IDs, not string labels).
      :type chain_iid: Tensor[int64], shape [batch, n_res]
      :param \*\*kwargs: Additional arguments passed to ``pose_stack_from_canonical_form``.
      
      :rtype: PoseStack
      
      :raises ValueError: If any ``residue_type`` value is outside 1..20 (protein-only).
      

.. py:function:: canonical_form_from_atomworks(coords: torch.Tensor, residue_type: torch.Tensor, chain_iid: torch.Tensor) -> tmol.io.canonical_form.CanonicalForm

   .. rubric:: Docstring

   .. code-block:: text

      Build a CanonicalForm from atomworks UNIFIED_ATOM37_ENCODING tensors.
      
      :param coords: Atom coordinates in the atomworks atom37 layout.
      :type coords: Tensor, shape [batch, n_res, 37, 3]
      :param residue_type: Atomworks token indices. Must be in 1..20 (standard protein only).
      :type residue_type: Tensor[int64], shape [batch, n_res]
      :param chain_iid: Chain identifiers.
      :type chain_iid: Tensor[int64], shape [batch, n_res]
      
      :rtype: CanonicalForm
      

.. py:function:: atomworks_from_pose_stack(pose_stack: tmol.pose.pose_stack.PoseStack) -> tuple

   .. rubric:: Docstring

   .. code-block:: text

      Convert a PoseStack back to atomworks UNIFIED_ATOM37_ENCODING tensors.
      
      :param pose_stack: The PoseStack to convert.  Must contain only standard amino acids.
      :type pose_stack: PoseStack
      
      :returns: * **coords** (*Tensor, shape [n_poses, max_n_res, 37, 3]*) -- Atom coordinates in the atomworks atom37 layout.  Absent atoms are 0.
                * **residue_type** (*Tensor[int64], shape [n_poses, max_n_res]*) -- Atomworks token indices (1..20 for real residues, 0 for padding).
                * **chain_iid** (*Tensor[int64], shape [n_poses, max_n_res]*) -- Chain identifiers.
      

.. py:function:: canonical_ordering_for_atomworks() -> tmol.io.canonical_ordering.CanonicalOrdering

   .. rubric:: Docstring

   .. code-block:: text

      Construct the CanonicalOrdering for the protein subset used
      by the atomworks UNIFIED_ATOM37_ENCODING.
      

.. py:function:: packed_block_types_for_atomworks(device: torch.device) -> tmol.pose.packed_block_types.PackedBlockTypes

   .. rubric:: Docstring

   .. code-block:: text

      Construct the PackedBlockTypes for the protein subset used
      by the atomworks UNIFIED_ATOM37_ENCODING.
      

