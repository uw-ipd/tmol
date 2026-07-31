tmol.pack.build_missing_sidechains
==================================

.. py:module:: tmol.pack.build_missing_sidechains


Functions
---------

.. autoapisummary::

   tmol.pack.build_missing_sidechains.build_missing_sidechains


Module Contents
---------------

.. py:function:: build_missing_sidechains(pose_stack: tmol.pose.pose_stack.PoseStack, sfxn: tmol.score.score_function.ScoreFunction, dunbrack_sampler: tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler.DunbrackChiSampler, block_has_missing_atoms: tmol.types.torch.Tensor[torch.bool][:, :], no_optH: bool = False) -> tmol.pose.pose_stack.PoseStack

   .. rubric:: Docstring

   .. code-block:: text

      Build missing sidechains and place hydrogens using per-block sampler assignment.
      
      Assigns samplers on a per-block basis in a single packing run:
      
      - Blocks with missing non-leaf (heavy) atoms: DunbrackChiSampler +
        FixedAAChiSampler.  The input conformation is not included as a rotamer
        because the sidechain is incomplete.
      - All other real blocks (leaf-only or no missing atoms): OptHSampler, which
        keeps heavy atoms fixed and samples proton chi angles and NHQ flips.
        FallbackSampler (always present by default) covers residue types that
        OptH does not handle (ALA, GLY, etc.).
      
      When no_optH=True the old behavior is preserved: only Dunbrack runs for
      blocks with missing heavy atoms; all other blocks are frozen.
      
      Note: IncludeCurrentSampler is intentionally not used.  For Dunbrack
      blocks the native conformation is broken and must not appear as a rotamer.
      For OptH blocks, OptH includes native as rotamer-0 for NHQ residues and
      FallbackSampler covers the rest.
      
      :param pose_stack: The pose stack to process.
      :param sfxn: Score function used for packing.
      :param dunbrack_sampler: DunbrackChiSampler configured from the parameter DB.
      :param block_has_missing_atoms: Boolean tensor [n_poses, max_n_blocks]; True
                                      for blocks that have missing non-leaf (heavy) atoms.
      :param no_optH: When True, skip OptH and preserve old Dunbrack-only behavior.
      
      :returns: PoseStack with missing sidechains built and (by default) hydrogens
                placed and optimized.
      

