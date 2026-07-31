tmol.score.genbonded.genbonded_energy_term
==========================================

.. py:module:: tmol.score.genbonded.genbonded_energy_term

.. rubric:: Module docstring

.. code-block:: text

   GenBonded energy term: torsional sub-term of cart-bonded using generic
   atom-type parameters rather than per-residue atom-name parameters.
   
   Key differences from CartBondedEnergyTerm:
     - Torsional interactions (proper torsions and improper torsions).
     - Parameter lookup is by atom chemical type (e.g. CS, CD, C*, X) rather
       than by (residue_name, atom_name).  The database carries a hierarchy that
       maps each concrete type to a sequence of fall-back types.
     - Intra-block parameters (proper and improper torsions) are resolved at
       setup_block_type time and stored as a single dense array, tagged by type
       (tag=0 proper, tag=1 improper).  Shape per entry: Vec<Int,5> for the
       4-atom subgraph + type tag, Vec<Real,5> for parameters.
     - Inter-block torsion parameters are stored in a hash table keyed by
       (type1, type2, type3, type4, bond_type_int) so that bond-type-specific
       entries are preferred over wildcard ('~') entries.
     - Bond type of the central bond is tracked through the pipeline and used
       for both intra (Python-time lookup) and inter (GPU-time hash lookup).
   


Attributes
----------

.. autoapisummary::

   tmol.score.genbonded.genbonded_energy_term.MAX_HIER_DEPTH
   tmol.score.genbonded.genbonded_energy_term.BOND_CHAR_TO_INT
   tmol.score.genbonded.genbonded_energy_term.BOND_TYPE_TO_CHAR
   tmol.score.genbonded.genbonded_energy_term.GB_WILDCARD_BOND_INT


Classes
-------

.. autoapisummary::

   tmol.score.genbonded.genbonded_energy_term.GenBondedEnergyTerm


Module Contents
---------------

.. py:data:: MAX_HIER_DEPTH
   :value: 3


.. py:data:: BOND_CHAR_TO_INT

.. py:data:: BOND_TYPE_TO_CHAR

.. py:data:: GB_WILDCARD_BOND_INT
   :value: 0


.. py:class:: GenBondedEnergyTerm(param_db: tmol.database.ParameterDatabase, device: torch.device)

   Bases: :py:obj:`tmol.score.atom_type_dependent_term.AtomTypeDependentTerm`


   .. py:attribute:: device
      :type:  torch.device


   .. py:attribute:: gen_database


   .. py:method:: class_name()
      :classmethod:



   .. py:method:: score_types()
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Return the list of score types that this EnergyTerm computes
         
         The order that the term reports score types in this function should be
         the same order that it reports the scores themselves in the output
         tensor
         


   .. py:method:: n_bodies()

      .. rubric:: Docstring

      .. code-block:: text

         Return the number of residues that this term operates on
         
         1, 2, or -1 to represent the whole structure
         


   .. py:method:: find_torsion_subgraphs(bonds)

      .. rubric:: Docstring

      .. code-block:: text

         Return list of (i, j, k, l) tuples for all proper torsions in *bonds*.
         
         Atoms are represented as local indices within the block.
         


   .. py:method:: find_improper_subgraphs(bonds)

      .. rubric:: Docstring

      .. code-block:: text

         Return list of (center, n1, n2, n3) tuples for all improper torsions.
         
         An atom is an improper center if it has exactly 3 bonded neighbors.
         The three neighbor indices are returned in sorted (canonical) order.
         


   .. py:method:: get_atom_chem_type(block_type: tmol.chemical.restypes.RefinedResidueType, atom_idx: int) -> str

      .. rubric:: Docstring

      .. code-block:: text

         Return the chemical atom type string for *atom_idx* in *block_type*.
         


   .. py:method:: resolve_torsion_params(block_type: tmol.chemical.restypes.RefinedResidueType, torsions)

      .. rubric:: Docstring

      .. code-block:: text

         For each torsion tuple (i,j,k,l), look up its genbonded parameters.
         
         Returns (kept_torsions, params) where:
           kept_torsions : filtered list of (i,j,k,l) tuples that had a DB match
           params        : numpy float32 array of shape (N_kept, 5):
                           columns [k1, k2, k3, k4, offset]
         
         Torsions with no matching database entry are dropped from the output.
         The central bond (j,k) bond type is looked up from block_type.bond_to_type
         and passed to find_torsion_params for bond-aware matching.
         


   .. py:method:: resolve_improper_params(block_type: tmol.chemical.restypes.RefinedResidueType, impropers)

      .. rubric:: Docstring

      .. code-block:: text

         For each improper tuple (center, n1, n2, n3), look up parameters.
         
         Returns (kept_impropers, params) where:
           kept_impropers : filtered list of tuples that had a DB match
           params         : numpy float32 array of shape (N_kept, 2): [k, delta]
         
         Impropers with no matching database entry are dropped.
         


   .. py:method:: atom_hierarchy_indices(atom_type: str) -> List[int]

      .. rubric:: Docstring

      .. code-block:: text

         Return a list of up to MAX_HIER_DEPTH type indices for *atom_type*.
         
         The list goes from most specific to most generic.  Padded with -1 to
         reach MAX_HIER_DEPTH elements.
         


   .. py:method:: setup_block_type(block_type: tmol.chemical.restypes.RefinedResidueType)

      .. rubric:: Docstring

      .. code-block:: text

         Make a one-time annotation on the block type. These annotations will
         probably require string comparison and may be slow; they should be
         performed only once, so the EnergyTerm must check that its annotation
         is not already present in the block type. Annotations should be in
         numpy data structures (and stored on the CPU).
         
         If the annotation requires more than one array, then the EnergyTerm
         should use a python class to store those arrays. E.g.,
         class FooSet:
             foo_array1: NDArray[numpy.int32][:]
             foo_array2: NDArray[numpy.int32][:, :]
         
         If the kind of annotation made depends on data that may change
         between different instances of the same term, then the annotation
         should be a map whose key is a function of the perhaps-changing
         data. The term should calculate that key at its construction to
         make retrieval efficient. (Any such data that sways how the
         calculation is made should never change over the lifetime of the
         instance; if new values for that data are needed a separate
         instance should be created.)
         


   .. py:method:: setup_packed_block_types(packed_block_types: tmol.pose.packed_block_types.PackedBlockTypes)

      .. rubric:: Docstring

      .. code-block:: text

         Make a one-time annotation of the packed-block types. This annotation
         should mostly involve concatenating the previously-made numpy annotations
         on the block types that the packed-block types contains. E.g. if the
         EnergyTerm annotates the block types with an i-dimensional array "foo,"
         then it should also annotate the PackedBlockTypes with an (i+1)-dimensional
         tensor "foo" where the first dimension will index across the different
         block types in foo in the order that those block types appear in the
         PackedBlockTypes' list of active block types. Sometimes the size of the
         i-dimensional arrays will differ between block types; the (i+1)-dimensional
         tensor should be dimensioned to the maximal size for each of the i dimensions
         among the set of dimensions of the various block types. The extra padding
         in such cases is recommended to be filled with a sentinel value of -1.
         
         As with the block type annotation, if more than one tensor is required,
         then the annotation should be a class. If the annotation is based on
         data that might differ between instances, then the annotation should be
         a map whose keys are determined by the data.
         
         The EnergyMethod should begin by checking that it has not already made
         this annotation. Any array data in the annotation should be torch
         tensors and should live on the PackedBlockTypes' device.
         


   .. py:method:: setup_poses(poses: tmol.pose.pose_stack.PoseStack)

      .. rubric:: Docstring

      .. code-block:: text

         Make a one-time annotation of a PoseStack. These annotations should
         not depend on anything about the conformation or block-type identity of
         the PoseStack, but can depend on the chemical connectivity, the number
         of poses in the stack, and the maximum number of atoms in the stack.
         
         Any array data should be stored in torch tensors and live on the
         pose_stack's device.
         


   .. py:method:: get_pose_score_term_function()


   .. py:method:: get_rotamer_score_term_function()


   .. py:method:: get_score_term_attributes(pose_stack: tmol.pose.pose_stack.PoseStack)


