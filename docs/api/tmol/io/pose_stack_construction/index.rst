tmol.io.pose_stack_construction
===============================

.. py:module:: tmol.io.pose_stack_construction


Functions
---------

.. autoapisummary::

   tmol.io.pose_stack_construction.pose_stack_from_canonical_form


Module Contents
---------------

.. py:function:: pose_stack_from_canonical_form(canonical_ordering: tmol.io.canonical_ordering.CanonicalOrdering, pbt: tmol.pose.packed_block_types.PackedBlockTypes, chain_id: tmol.types.torch.Tensor[torch.int32][:, :], res_types: tmol.types.torch.Tensor[torch.int32][:, :], coords: tmol.types.torch.Tensor[torch.float32][:, :, :, 3], res_labels: Optional[tmol.types.array.NDArray[int][:, :]], res_ins_codes: Optional[tmol.types.array.NDArray[object][:, :]], chain_labels: Optional[tmol.types.array.NDArray[object][:, :]], atom_occupancy: Optional[tmol.types.array.NDArray[numpy.float32][:, :, :]] = None, atom_b_factor: Optional[tmol.types.array.NDArray[numpy.float32][:, :, :]] = None, disulfides: Optional[tmol.types.torch.Tensor[torch.int64][:, 3]] = None, res_not_connected: Optional[tmol.types.torch.Tensor[torch.bool][:, :, 2]] = None, *, find_additional_disulfides: Optional[bool] = True, return_chain_ind: bool = False, return_atom_mapping: bool = False, return_block_has_missing_atoms: bool = False)

   .. rubric:: Docstring

   .. code-block:: text

      Create a PoseStack, resolving which block type is requested by the
      presence and absence of the provided atoms for each residue type.
      There are five required arguments and several optional arguments.
      
      Arguments:
      canonical_ordering: an object that describes the set of atoms that each
          residue type (aka block type) and all of its interchangable variants
          contain and the order in which those atoms should appear in the
          coords tensor
      packed_block_types: the object that holds score-term annotations needed
          by the score terms and which is intended to be shared between
          multiple PoseStacks for efficiency; the PoseStack this function
          creates will hold this packed_block_types object
      chain_id: an n-pose x max-n-residue tensor with an index for which chain
          each residue belongs to. Residues belonging to the same chain must be
          consecutive.
      res_types: an n-pose x max-n-residue tensor with the canonically
          ordered amino acid index for each residue. A sentinel value of "-1"
          should be used to indicate that a given position does not contain
          a residue (perhaps because it belongs to a pose with fewer than
          max-n-residue residues; each pose in the PoseStack is allowed to
          have fewer than max-n-residue residues).
      coords: an n-pose x max-n-residue x max-n-atoms-per-residue tensor
          providing the coordinates of some or all of the atoms. The order
          in which atoms should appear in this tensor is given by the
          CanonicalOrdering object. Any atoms whose coordinates are not
          being provided to this function must have their coordinates marked
          as NaN. Any atom with a coordinate of NaN will be taken as "not
          present" and all others (including any coordinates at the origin, e.g.)
          will be treated as if they "are present." Note: "present" here
          means "the coordinate is being provided" and not "this atom
          should be modeled;" conversely, there is no way to say "do not
          include a particular atom in tmol calculations."
      
          Currently, all heavy atoms must be provided to tmol except
          leaf atoms. A "leaf atom" is one that has no atoms that use it
          as a parent or grand parent when describing their icoors.
          Hydrogen atoms are all leaf atoms. Backbone carbonyl oxygens
          are also leaf atoms. Even though hydrogen atoms are optional,
          the hydroxyl hydrogens on SER, THR, and TYR are recommended
          as tmol will build them suboptimally: at a dihedral of 180
          regardless of the presence of nearby hydrogen-bond acceptors
      
      disulfides: an optional n-total-disulfides x 3 tensor. If this argument
          is not provided, then the coordinates of SG atoms on CYS residues
          will be used to determine which pairs are closest to each other and
          within a cutoff distance of 2.5A and declare those SGs as forming
          disulfide bonds. This means that SGs slightly longer than 2.5A
          will not be detected. If you should know which pairs of cysteines
          form disulfide bonds, then you can provide their pairing:
          [ [pose_ind, cys1_ind, cys2_ind], ...].
      
      find_additional_disulfides: an optional boolean argument to control whether
          to look for disulfide bonds between pairs of CYS residues that are
          not listed in the "disulfides" argument. By default, this is True,
          but if you want to skip disulfide detection or want to prevent
          unpaired CYS from being locked into disulfides, then set this flag
          to False
      
      res_not_connected: an optional input used to indicate that a given (polymeric)
          residue is not connected to either its previous or next residue; for
          termini residues, they will not be built with their termini-variant
          types. The purpose is to allow the user to include a subset of the
          residues in a protein where a series of "gap" residues can be omitted
          between i and i+1 without those two residues being treated as if they
          are chemically bonded. This will keep the Ramachandran term from scoring
          nonsense dihdral angles and will keep the cart-bonded term from scoring
          nonsense bond lengths and angles.
      
      
      Optional return values:
          If any of the following flags are provided, this function will return a tuple
          instead of just a pose stack, with the first argument being the pose stack and
          the second argument being a dictionary with keys corresponding to the requested
          values.
      
          return_chain_ind: return the chain-index tensor as "chain_ind" that has been
              "left-justified" from the chain
      
          return_atom_mapping: return the mapping for atoms in the canonical-form tensor
              to their PoseStack index; this could be used to update the coordinates
              in a PoseStack without rebuilding it (as long as the chemical identity
              is meant to be unchanged) or to perhaps remap derivatives to or from
              pose stack ordering. If requested, the atom mapping will returned as two tensors
              under the keys "ps_atom_mapping" and "can_atom_mapping" by this function:
                  ps, opt_vals = pose_stack_from_canonical_form(
                      ...,
                      return_atom_mapping=True
                  )
                  t1 = opt_vals["can_atom_mapping"]
                  t2 = opt_vals["ps_atom_mapping"]
                  can_ord_coords[
                      t1[:, 0], t1[:, 1], t1[:, 2]
                  ] = ps.coords[
                      t2[:, 0], t2[:, 1]
                  ]
              where can_atom_mapping is a tensor nats x 3 where
              - position [i, 0] is the pose index
              - position [i, 1] is the residue index, and
              - position [i, 2] is the canonical-ordering atom index
              and ps_atom_mapping is a tensor nats x 2 where
              - position [i, 0] is the pose index, and
              - position [i, 1] is the pose-ordered atom index
      
          return_block_has_missing_atoms: returns a [n_pose x max_n_res] bool
              tensor in the dictionary under the key "block_has_missing_atoms" with
              elements being true iff any non-leaf atoms were missing (NaN). To be used
              with a packer to build these missing atoms. If this argument is False, an
              exception will be thrown when these missing atoms are encountered.
      

