tmol.io.canonical_ordering
==========================

.. py:module:: tmol.io.canonical_ordering


Classes
-------

.. autoapisummary::

   tmol.io.canonical_ordering.ordered_set
   tmol.io.canonical_ordering.CysSpecialCaseIndices
   tmol.io.canonical_ordering.HisSpecialCaseIndices
   tmol.io.canonical_ordering.CanonicalOrdering


Functions
---------

.. autoapisummary::

   tmol.io.canonical_ordering.default_canonical_ordering
   tmol.io.canonical_ordering.default_packed_block_types
   tmol.io.canonical_ordering.canonical_form_from_pdb
   tmol.io.canonical_ordering.select_atom_records_res_subset
   tmol.io.canonical_ordering.canonical_form_from_atom_records


Module Contents
---------------

.. py:class:: ordered_set(input_values=None)

   .. py:attribute:: ordered_vals
      :value: []



   .. py:attribute:: unordered_vals


   .. py:method:: add(val)


.. py:class:: CysSpecialCaseIndices

   .. py:attribute:: cys_co_aa_ind
      :type:  int


   .. py:attribute:: sg_atom_for_co_cys
      :type:  int


.. py:class:: HisSpecialCaseIndices

   .. py:attribute:: his_co_aa_ind
      :type:  int


   .. py:attribute:: his_ND1_in_co
      :type:  int


   .. py:attribute:: his_NE2_in_co
      :type:  int


   .. py:attribute:: his_HD1_in_co
      :type:  int


   .. py:attribute:: his_HE2_in_co
      :type:  int


   .. py:attribute:: his_HN_in_co
      :type:  int


   .. py:attribute:: his_NH_in_co
      :type:  int


   .. py:attribute:: his_NN_in_co
      :type:  int


   .. py:attribute:: his_CG_in_co
      :type:  int


.. py:class:: CanonicalOrdering

   .. rubric:: Docstring

   .. code-block:: text

      The canonical ordering class describes the integer ordering
      of residue types and for atoms within those residue types
      for the collection of available residue types defined by a
      PatchedChemicalDatabase.
      
      The canonical ordering class's purpose is to enable creation of
      a "canonical form" dictionary that describes a molecular system
      in the way that tmol expects in order to construct a PoseStack.
      
      There is no "canonical form" dictionary is simply a dictionary
      holding the at-least-three-but-as-many-as-eight arguments to
      tmol.io.pose_stack_construction.pose_stack_from_canonical_form
      after the first two. That is, it must contain "chain_id",
      "res_types" and "coords" entries.
      
      When constructing a PoseStack, there are multiple residue types
      for each "equivalence class" (think 3-letter code); e.g. for
      "CYS" there's the standard middle-of-a-polypeptide-chain CYS,
      the standard middle-of-a-polypeptide-chain disulfide-forming CYS,
      and then for those two, four variants for the N-, C-, and both-N-
      and-C terminal forms; eight total options for a single "CYS"
      three-letter code. tmol collects all of the various forms of
      a single equivalence class and creates a list of all atom names
      across all the residue types for it. You can then provide tmol
      the set of atoms that are present at a given position by giving
      a non-NaN coordinate for that entry in an
      [n-poses x max-n-res x max-ats-per-res x 3] tensor of
      coordinates. Atoms with NaN coordinates are taken as possibly
      present in the residue type; tmol will decide the best fit
      for which residue type to use at each position.
      If an atom is provided to tmol and it is not present for a
      given residue type, then that residue type will be disqualified
      from consideration. Thus an important part of telling
      tmol which atoms are present is mapping from an atom name to
      an index for that atom. The CanonicalOrdering object is where
      that mapping is encoded. It also handles the mapping from
      alternate-atom-name to canonical-form-atom index; e.g. in
      PDBv2, glycine's two hydrogens were named "HA1" and "HA2",
      but in PDBv3, they are named "1HA" and "2HA." So that we can
      parse PDB files written in PDBv2 and PDBv3, we have an idea of
      an "alias" for an atom; see the restypes_atom_index_mapping
      data member.
      
      There are four data members that are useful for users:
          - max_n_canonical_atoms
          - restype_io_equiv_classes
          - restypes_ordered_atom_names
          - restypes_atom_index_mapping
      the remaining data members are useful primarily for
      internal tmol functionality
      
      max_n_canonical_atoms: the largest number of distinct atom names among all
          variants of a single residue type (equivalence class) across all residue types
      
      restype_io_equiv_classes:
          essentially the list of 3-letter codes for the residue
          types that are readable; use the index function
          (e.g. co.restype_io_equiv_classes.index("TRP"))
          to obtain the integer meant to represent each restype
      
      restypes_ordered_atom_names:
          the ordered list of the names of each atom for every allowed
          residue type; does not include the alternate names for atoms.
          Atoms should be given to tmol in this order; e.g. by putting
          the coordinate of the ith atom in the ith entry of the
          coordinate tensor (e.g. coords[p, r, i] for pose p, residue r)
      
      restypes_atom_index_mapping:
          mapping for each name3 from atom name and atom name alias
          to the index of that atom for every allowed residue
          type in the restypes_ordered_atom_names list; this is
          probably more useful than the restypes_ordered_atom_names
          list, especially if you are using the PDBv2 naming
          convention (as Rosetta3 does) instead of the PDBv3
          convention.
      

   .. py:attribute:: max_n_canonical_atoms
      :type:  int


   .. py:attribute:: restype_io_equiv_classes
      :type:  Tuple[str, Ellipsis]


   .. py:attribute:: restypes_ordered_atom_names
      :type:  Mapping[str, Tuple[str, Ellipsis]]


   .. py:attribute:: restypes_atom_index_mapping
      :type:  Mapping[str, Mapping[str, int]]


   .. py:attribute:: restypes_mainchain_atoms
      :type:  Mapping[str, Optional[Tuple[str, Ellipsis]]]


   .. py:attribute:: restypes_default_termini_mapping
      :type:  Mapping[str, Tuple[str, str]]


   .. py:attribute:: down_termini_patches
      :type:  Tuple[str, Ellipsis]


   .. py:attribute:: up_termini_patches
      :type:  Tuple[str, Ellipsis]


   .. py:attribute:: termini_patch_added_atoms
      :type:  Mapping[str, Tuple[str, Ellipsis]]


   .. py:attribute:: cys_inds
      :type:  CysSpecialCaseIndices


   .. py:attribute:: his_inds
      :type:  HisSpecialCaseIndices


   .. py:property:: n_restype_io_equiv_classes


   .. py:method:: extra_atoms()
      :classmethod:



   .. py:method:: from_chemdb(chemdb: tmol.chemical.patched_chemdb.PatchedChemicalDatabase)
      :classmethod:



   .. py:method:: create_src_2_tmol_mappings(src_aa_name3s, src_atom_names_for_name3s, device)


.. py:function:: default_canonical_ordering() -> CanonicalOrdering

   .. rubric:: Docstring

   .. code-block:: text

      Create a CanonicalOrdering object from the default set of residue types
      

.. py:function:: default_packed_block_types(device: torch.device) -> tmol.pose.packed_block_types.PackedBlockTypes

   .. rubric:: Docstring

   .. code-block:: text

      Create a PackedBlockTypes object from the default set of residue types
      

.. py:function:: canonical_form_from_pdb(canonical_ordering: CanonicalOrdering, pdb_lines_or_fname: Union[str, List], device: torch.device, *, residue_start: Optional[int] = None, residue_end: Optional[int] = None, res_not_connected: Optional[tmol.types.torch.Tensor[torch.bool][:, :, 2]] = None) -> tmol.io.canonical_form.CanonicalForm

   .. rubric:: Docstring

   .. code-block:: text

      Create a canonical form from either the contents of a PDB file
      as one long string or a list of individual lines from the file or
      by providing the name/path of a PDB file
      
      pdb_lines_or_fname must either be a list of the lines in a PDB file or
      a string representing a file
      
      

.. py:function:: select_atom_records_res_subset(atom_records: pandas.DataFrame, residue_start: Optional[int], residue_end: Optional[int])

   .. rubric:: Docstring

   .. code-block:: text

      Figure out the starting row index for each residue
      and take the slice of the atom_records dataframe containing
      every atom of every residue within the given inclusive range.
      If either residue_start or residue_end are omitted, then
      the are treated as being the first or last residue.
      

.. py:function:: canonical_form_from_atom_records(canonical_ordering: CanonicalOrdering, atom_records: pandas.DataFrame, device: torch.device, res_not_connected: Optional[tmol.types.torch.Tensor] = None) -> tmol.io.canonical_form.CanonicalForm

