tmol.io.pose_stack_from_biotite
===============================

.. py:module:: tmol.io.pose_stack_from_biotite


Attributes
----------

.. autoapisummary::

   tmol.io.pose_stack_from_biotite.logger


Classes
-------

.. autoapisummary::

   tmol.io.pose_stack_from_biotite.BiotitePoseBuildContext


Functions
---------

.. autoapisummary::

   tmol.io.pose_stack_from_biotite.build_context_from_biotite
   tmol.io.pose_stack_from_biotite.pose_stack_from_biotite
   tmol.io.pose_stack_from_biotite.biotite_from_pose_stack
   tmol.io.pose_stack_from_biotite.canonical_form_from_biotite
   tmol.io.pose_stack_from_biotite.canonical_ordering_for_biotite
   tmol.io.pose_stack_from_biotite.packed_block_types_for_biotite
   tmol.io.pose_stack_from_biotite.get_element_from_atom_name
   tmol.io.pose_stack_from_biotite.biotite_from_canonical_form


Module Contents
---------------

.. py:data:: logger

.. py:class:: BiotitePoseBuildContext

   .. rubric:: Docstring

   .. code-block:: text

      Immutable, structure-independent construction context.
      
      Holds only the pieces that depend on the parameter database / ligand set
      (not on any particular input structure), so it can be built once and reused
      across many structures that share the same ligand(s). The per-structure
      canonical form is computed separately by ``pose_stack_from_biotite`` for
      each input structure.
      

   .. py:attribute:: canonical_ordering
      :type:  tmol.io.canonical_ordering.CanonicalOrdering


   .. py:attribute:: packed_block_types
      :type:  tmol.pose.packed_block_types.PackedBlockTypes


   .. py:attribute:: parameter_database
      :type:  tmol.database.ParameterDatabase


   .. py:attribute:: restype_set
      :type:  tmol.chemical.restypes.ResidueTypeSet


   .. py:attribute:: fragment_definitions
      :type:  tuple
      :value: ()



.. py:function:: build_context_from_biotite(biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack, torch_device: torch.device, param_db: tmol.database.ParameterDatabase | None = None, prepare_ligands: bool = False, ligand_ph: float = 7.4, strict_atom_types: bool = False, strict_ligands: bool = True, ligand_params_files: list[str] | None = None, sample_proton_chi: bool = True) -> BiotitePoseBuildContext

   .. rubric:: Docstring

   .. code-block:: text

      Build the structure-independent construction context.
      
      The returned context holds only database/ligand-derived pieces (canonical
      ordering, residue-type set, packed block types, parameter database); it does
      not depend on the input structure's coordinates and can be reused across
      structures sharing the same ligand(s). ``biotite_structure`` is used only to
      detect and prepare ligands (when ``prepare_ligands=True``).
      
      :param biotite_structure: Input AtomArray or AtomArrayStack. Used only for
                                ligand detection/preparation when ``prepare_ligands=True``.
      :param torch_device: Target torch device.
      :param param_db: Optional parameter database. When provided, canonical ordering,
                       residue types, and packed block types are built from this database.
                       If prepare_ligands=True, it is extended with ligand data. If None,
                       defaults are used.
      :param prepare_ligands: If True, detect and prepare non-standard residues
                              (via ``tmol.ligand``, which uses RDKit for atom typing and
                              residue-type construction).
      :param ligand_ph: Target pH for ligand protonation (default 7.4, only used when
                        prepare_ligands=True).
      :param strict_atom_types: If True, unknown ligand atom types raise errors
                                instead of using a fallback element heuristic.
      :param strict_ligands: If True (default), raise when a detected ligand cannot
                             be prepared and registered (instead of silently dropping it during
                             pose construction). Pass False to fall back to warn-and-skip. Only
                             used when prepare_ligands=True.
      :param ligand_params_files: Optional list of tmol YAML params file paths.
                                  Residues defined in these files skip the RDKit/OB pipeline.
      :param sample_proton_chi: If True, prepared ligands emit PROTON_CHI
                                ``chi_samples`` for polar-hydrogen rotations (driving OptHSampler).
                                Enabled by default; pass False to suppress proton-chi samples. Only
                                used when prepare_ligands=True.
      
      :returns: BiotitePoseBuildContext containing canonical ordering, packed block
                types, parameter database, and residue type set.
      

.. py:function:: pose_stack_from_biotite(biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack, torch_device: torch.device, param_db: tmol.database.ParameterDatabase | None = None, missing_density_distance_threshold: float = 2.4, no_optH: bool = False, prepare_ligands: bool = False, ligand_ph: float = 7.4, strict_atom_types: bool = False, strict_ligands: bool = True, ligand_params_files: list[str] | None = None, sample_proton_chi: bool = True, return_context: bool = False, context: BiotitePoseBuildContext | None = None, **kwargs: object) -> tmol.pose.pose_stack.PoseStack | tuple[tmol.pose.pose_stack.PoseStack, dict] | tuple[tmol.pose.pose_stack.PoseStack, BiotitePoseBuildContext]

   .. rubric:: Docstring

   .. code-block:: text

      Build a PoseStack from the output generated by Biotite.
      
      To score many structures that share the same ligand(s) efficiently, build
      the (expensive, structure-independent) context once and reuse it::
      
          context = build_context_from_biotite(struct0, dev, prepare_ligands=True)
          for struct in structures:
              pose_stack = pose_stack_from_biotite(struct, dev, context=context)
      
      Reusing a context skips rebuilding the parameter database, canonical
      ordering, residue-type set, and packed block types; only the per-structure
      canonical form is recomputed (see the ``context`` arg).
      
      :param biotite_structure: A Biotite AtomArray or AtomArrayStack.
      :param torch_device: Target PyTorch device.
      :param param_db: Optional ParameterDatabase. When provided, conversion and pose
                       construction use this database. If prepare_ligands=True, it is
                       extended with ligand data. Mutually exclusive with ``context``.
      :param missing_density_distance_threshold: Distance threshold in Angstroms.
                                                 Adjacent residues whose closest inter-atom distance exceeds this
                                                 value are treated as disconnected (upper/lower connects broken).
                                                 Set to 0 to disable. Default is 2.4.
      :param no_optH: When False (default), all residues with complete heavy atoms
                      are packed with OptHSampler to place and optimize hydrogen positions
                      and NHQ flips, while residues with missing heavy atoms are rebuilt
                      with DunbrackChiSampler.  When True, only missing heavy-atom
                      sidechains are rebuilt with Dunbrack; hydrogens are left at the
                      kinematically ideal positions produced during pose construction.
      :param prepare_ligands: If True, detect and prepare non-standard residues
                              (see ``build_context_from_biotite`` for details).
      :param ligand_ph: Target pH for ligand protonation (default 7.4, only used when
                        prepare_ligands=True).
      :param strict_atom_types: If True, unknown ligand atom types raise errors
                                instead of using a fallback element heuristic.
      :param strict_ligands: If True (default), raise when a detected ligand cannot
                             be prepared and registered, instead of silently dropping it. Pass
                             False to warn-and-skip. Only used when prepare_ligands=True.
      :param ligand_params_files: Optional list of tmol YAML params file paths.
      :param sample_proton_chi: If True, prepared ligands emit PROTON_CHI
                                ``chi_samples`` so OptHSampler samples ligand polar-H rotamers
                                (enabled by default; pass False to disable). Only used when
                                prepare_ligands=True.
      :param return_context: If True, return ``(pose_stack, BiotitePoseBuildContext)``.
      :param \*\*kwargs: Additional arguments passed to pose_stack_from_canonical_form.
      
      :returns: PoseStack when no optional values requested and return_context is False.
                ``(PoseStack, BiotitePoseBuildContext)`` when return_context is True.
                ``(PoseStack, dict)`` when optional return values were requested via kwargs.
                Fragmented poses expose their block mapping as
                ``pose_stack.fragmented_ligand_mapping``.
      

.. py:function:: biotite_from_pose_stack(pose_stack: tmol.pose.pose_stack.PoseStack, co: tmol.io.canonical_ordering.CanonicalOrdering | None = None, merge_fragments: bool = True) -> biotite.structure.AtomArray | biotite.structure.AtomArrayStack

   .. rubric:: Docstring

   .. code-block:: text

      Convert PoseStack back to Biotite structure.
      
      :param pose_stack: Pose stack to convert.
      :param co: Canonical ordering used for conversion. Provide the ordering that
                 was used when ligands or custom residue types are present.
      :param merge_fragments: Restore fragmented ligands to their original residue
                              identity. Set to False to keep fragment residues separate.
      
      :returns: Biotite AtomArray for single-pose or AtomArrayStack for multi-pose.
      

.. py:function:: canonical_form_from_biotite(biotite_structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack, torch_device: torch.device, co: tmol.io.canonical_ordering.CanonicalOrdering | None = None, missing_density_distance_threshold: float = 2.4) -> tmol.io.canonical_form.CanonicalForm

   .. rubric:: Docstring

   .. code-block:: text

      Convert a Biotite AtomArray or AtomArrayStack to a CanonicalForm.
      
      This function bridges between Biotite's data structures and tmol's internal
      representation by converting atom and residue information from string-based
      identifiers to tmol's canonical integer-based indexing system.
      
      :param biotite_structure: A Biotite AtomArray (single structure) or
                                AtomArrayStack (multiple structures) containing the molecular data.
                                Must contain atom coordinates, residue names, atom names, chain IDs,
                                and optionally B-factors and occupancy values.
      :param torch_device: PyTorch device (e.g., torch.device('cuda') or torch.device('cpu'))
                           where the resulting tensors should be allocated.
      :param co: A CanonicalForm in case you want to use a non-default database (and thus may need
                 a different mapping)
      
      :returns:
      
                A data structure containing:
                    - chain_id: Tensor mapping residues to chain indices
                    - res_types: Tensor mapping residues to tmol residue type indices
                    - coords: 4D tensor of atomic coordinates (poses x residues x atoms x 3)
                    - res_labels: Original residue sequence numbers from the structure
                    - residue_insertion_codes: PDB insertion codes for residues
                    - chain_labels: Original chain identifiers from the structure
                    - atom_occupancy: Optional tensor of atom occupancy values
                    - atom_b_factor: Optional tensor of atom B-factor values
                    - disulfides: None (not handled in this conversion)
                    - res_not_connected: Tensor describing whether two consecutive residues
                      should be treated as chemically bonded.
      :rtype: CanonicalForm
      

.. py:function:: canonical_ordering_for_biotite() -> tmol.io.canonical_ordering.CanonicalOrdering

   .. rubric:: Docstring

   .. code-block:: text

      Construct the CanonicalOrdering object to use for Biotite.
      This wont be used as a typical CanonicalOrdering object, since
      we aren't mapping from int-to-int, and instead are going from
      string-to-int.
      

.. py:function:: packed_block_types_for_biotite(device: torch.device) -> tmol.pose.packed_block_types.PackedBlockTypes

   .. rubric:: Docstring

   .. code-block:: text

      Construct the PackedBlockTypes (PBT) object that will used for Biotite.
      We'll use the defaults since anything might show up in a Biotite AtomArray.
      Some things may show up in the AtomArrays that are not handled by this
      PBT, but that is work for the future.
      

.. py:function:: get_element_from_atom_name(atom_name: str) -> str

   .. rubric:: Docstring

   .. code-block:: text

      Parses a 4-character PDB atom name to return the element symbol.
      PDB columns 13-16:
      - Elements with 2 letters start at col 13.
      - Elements with 1 letter start at col 14.
      

.. py:function:: biotite_from_canonical_form(cf: tmol.io.canonical_form.CanonicalForm, co: tmol.io.canonical_ordering.CanonicalOrdering | None = None) -> biotite.structure.AtomArray | biotite.structure.AtomArrayStack

