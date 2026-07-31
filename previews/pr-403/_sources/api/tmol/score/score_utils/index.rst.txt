tmol.score.score_utils
======================

.. py:module:: tmol.score.score_utils


Classes
-------

.. autoapisummary::

   tmol.score.score_utils.FragmentInteractionScores


Functions
---------

.. autoapisummary::

   tmol.score.score_utils.calculate_fragment_interactions
   tmol.score.score_utils.residue_mask_from_chain
   tmol.score.score_utils.calculate_block_pair_ddg
   tmol.score.score_utils.res_mask_to_coord_mask
   tmol.score.score_utils.build_sidechain_coord_mask
   tmol.score.score_utils.compute_block_centroids_and_furthest_dist
   tmol.score.score_utils.build_coord_mask_for_mask_and_interacting_atoms
   tmol.score.score_utils.build_coord_mask_for_mask_and_nearby_blocks
   tmol.score.score_utils.compute_block_adjacency


Module Contents
---------------

.. py:class:: FragmentInteractionScores

   .. rubric:: Docstring

   .. code-block:: text

      Per-fragment interactions with an explicitly selected partner.
      

   .. py:attribute:: scores
      :type:  torch.Tensor


   .. py:attribute:: mapping
      :type:  tuple


.. py:function:: calculate_fragment_interactions(pose_stack, partner_mask, *, sfxn, mapping=None, sum_terms=False)

   .. rubric:: Docstring

   .. code-block:: text

      Return each ligand fragment's interaction with ``partner_mask``.
      
      The connected multi-block pose is scored once. Fragment-fragment entries
      remain in the block-pair matrix and are not silently assigned to either
      fragment.
      
      ``sfxn`` is required and must be built from the same ligand-extended
      parameter database used to construct ``pose_stack``.
      
      :returns: :class:`FragmentInteractionScores`. ``scores`` has shape
                ``[n_terms, n_poses, n_fragments]`` or ``[n_poses, n_fragments]`` when
                ``sum_terms`` is true.
      

.. py:function:: residue_mask_from_chain(pose_stack, chain_id)

   .. rubric:: Docstring

   .. code-block:: text

      Build a boolean block-level mask selecting residues belonging to a given
      chain, identified by its PDB chain label string.
      
      :param pose_stack: The pose stack. Must have a ``pdb_info`` attribute with
                         ``chain_labels`` (``NDArray[object][n_poses, max_n_blocks]``).
      :param chain_id: A string chain identifier (e.g. ``"A"``, ``"B"``).
      
      :returns: Boolean tensor of shape ``[n_poses, n_blocks]`` with ``True`` for
                residues whose PDB chain label matches ``chain_id``.
      

.. py:function:: calculate_block_pair_ddg(pose_stack, mask, mask2=None, sfxn=None, sum_terms=True, minimize=True, pack=False, database=None, return_pose_stack=False)

   .. rubric:: Docstring

   .. code-block:: text

      Calculate DDG score between two subsets of blocks within each pose, defined by 2 masks.
      If only one mask is provided, it will use the inverse of the first mask for the second.
      
      :param pose_stack: The pose stack to score
      :param mask: Boolean tensor of shape [n_poses, n_blocks]. True values indicate masked indices.
      :param mask2: Boolean tensor of shape [n_poses, n_blocks]. If not provided, it will use the inverse
                    of the first mask as the second mask.
      :param sfxn: Optional score function to use. If not provided, will default to beta2016
      :param sum_terms: If True, sum all score terms into a single score per pose. If False,
                        return per-term scores.
      :param minimize: If True (default), run cartesian minimization on the masked atoms before
                       computing the DDG score.
      :param pack: If True, pack (repack) rotamers of residues in the mask and residues adjacent
                   to the mask (computed via ``compute_block_adjacency``) before the minimization step.
      :param return_pose_stack: If True, also return the (possibly packed/minimized) pose stack
                                that was actually scored, as ``(ddg_scores, pose_stack)``.
      
      :returns: Tensor of shape [n_poses] or [n_terms, n_poses] containing the ddg score for each pose,
                separated by terms if requested. If ``return_pose_stack`` is True, returns a tuple
                ``(ddg_scores, pose_stack)``.
      

.. py:function:: res_mask_to_coord_mask(pose_stack, mask)

   .. rubric:: Docstring

   .. code-block:: text

      Convert a block-level (residue) boolean mask to an atom-level coordinate mask.
      
      For each pose, atoms belonging to blocks where the mask is True are marked as True
      in the output. The output mask can be used as a ``coord_mask`` argument to
      functions like ``run_cart_min``.
      
      :param pose_stack: The pose stack. Must have attributes ``coords``, ``max_n_blocks``,
                         ``max_n_block_atoms``, ``block_coord_offset64``, and ``real_atoms``.
      :param mask: Boolean tensor of shape ``[n_poses, n_blocks]``. ``True`` at
                   ``mask[i, j]`` indicates that all atoms of block ``j`` in pose ``i``
                   should be marked.
      
      :returns: Boolean tensor of shape ``[n_poses, max_n_atoms_per_pose]``, where
                ``max_n_atoms_per_pose = pose_stack.coords.shape[1]``.
      

.. py:function:: build_sidechain_coord_mask(pose_stack)

   .. rubric:: Docstring

   .. code-block:: text

      Build a coord_mask that selects only atoms belonging to sidechains.
      
      For polymeric residues, sidechain atoms are defined as real atoms that are
      NOT mainchain atoms. For non-polymeric residues, all real atoms are
      considered sidechain atoms. The output mask can be used as a ``coord_mask``
      argument to functions like ``run_cart_min``.
      
      :param pose_stack: The pose stack. Must have attributes ``coords``,
                         ``max_n_blocks``, ``max_n_block_atoms``, ``block_coord_offset64``,
                         ``real_atoms``, ``block_type_ind64``, and ``packed_block_types``.
      
      :returns: Boolean tensor of shape ``[n_poses, max_n_atoms_per_pose]``, where
                ``max_n_atoms_per_pose = pose_stack.coords.shape[1]``. True at
                ``coord_mask[i, j]`` indicates that atom ``j`` of pose ``i`` is a
                sidechain atom.
      

.. py:function:: compute_block_centroids_and_furthest_dist(pose_stack)

   .. rubric:: Docstring

   .. code-block:: text

      For each block in a pose stack, compute the average coordinate (centroid)
      of all of its atoms, as well as the distance of the furthest atom from this
      average.
      
      :param pose_stack: A PoseStack object.
      
      :returns:
      
                Tensor of shape [n_poses, n_blocks, 3] containing
                    the average coordinate (centroid) of all real atoms in each block.
                    Padding blocks and blocks with no atoms will have NaN centroids.
                block_furthest_dist: Tensor of shape [n_poses, n_blocks] containing
                    the distance of the furthest atom from the centroid for each block.
                    Padding blocks and blocks with no atoms will have NaN values.
      :rtype: block_centroids
      

.. py:function:: build_coord_mask_for_mask_and_interacting_atoms(pose_stack, mask)

.. py:function:: build_coord_mask_for_mask_and_nearby_blocks(pose_stack, mask)

   .. rubric:: Docstring

   .. code-block:: text

      Build a coord mask starting from a per-block mask, extending to
      sidechain atoms of blocks whose centroid is within dynamic range of
      any masked block centroid.
      
      All atoms from blocks in ``mask`` are unconditionally included.
      Additionally, sidechain atoms from any *unmasked* block are included
      if the distance between its centroid and the centroid of **any** masked
      block is **less than the sum of their respective furthest-atom-from-centroid
      distances**.
      
      :param pose_stack: The pose stack. Must have attributes ``coords``,
                         ``max_n_blocks``, ``max_n_block_atoms``, ``block_coord_offset64``,
                         ``real_atoms``, ``block_type_ind64``, ``block_type_ind``, and
                         ``packed_block_types``.
      :param mask: Boolean tensor of shape ``[n_poses, n_blocks]``.
      
      :returns: Boolean tensor of shape ``[n_poses, max_n_atoms]`` suitable for
                use as a ``coord_mask`` argument to ``run_cart_min``.
      

.. py:function:: compute_block_adjacency(block_centroids, block_furthest_dist, constant=5.0)

   .. rubric:: Docstring

   .. code-block:: text

      Compute a boolean block-level adjacency matrix.
      
      Two blocks *i* and *j* (in the same pose) are considered adjacent when
      the distance between their centroids is **less than the sum of their
      furthest-atom-from-centroid distances plus a constant**.
      
      .. math::
      
          \|\mathbf{c}_i - \mathbf{c}_j\|
          < d_i + d_j + \text{constant}
      
      :param block_centroids: Tensor of shape ``[n_poses, n_blocks, 3]``
                              containing the centroid coordinate of each block (e.g. as
                              returned by :func:`compute_block_centroids_and_furthest_dist`).
      :param block_furthest_dist: Tensor of shape ``[n_poses, n_blocks]``
                                  containing the distance of the atom furthest from the centroid
                                  for each block.
      :param constant: A scalar added to the sum of furthest distances.  Default
                       is ``5.0``.
      
      :returns: Boolean tensor of shape ``[n_poses, n_blocks, n_blocks]`` where
                ``adjacency[p, i, j]`` is ``True`` when the two blocks *i* and *j*
                in pose *p* are adjacent.
      
                The diagonal is always ``False`` (a block is not adjacent to itself).
                Padding / NaN-containing blocks are treated as not adjacent to any
                block.
      

