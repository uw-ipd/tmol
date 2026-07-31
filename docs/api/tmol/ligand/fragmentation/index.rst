tmol.ligand.fragmentation
=========================

.. py:module:: tmol.ligand.fragmentation

.. rubric:: Module docstring

.. code-block:: text

   User-defined fragmentation of fully prepared ligands.
   
   Fragments are specified by the integer ``tmol_fragment_id`` annotation on the
   input Biotite AtomArray.  Chemistry is perceived once for the complete ligand;
   the functions in this module only partition that prepared chemistry.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.fragmentation.FRAGMENT_ID_ANNOTATION
   tmol.ligand.fragmentation.MAX_FRAGMENT_CONNECTIONS
   tmol.ligand.fragmentation.MIN_FRAGMENT_HEAVY_ATOMS


Classes
-------

.. autoapisummary::

   tmol.ligand.fragmentation.FragmentConnection
   tmol.ligand.fragmentation.LigandFragmentDefinition
   tmol.ligand.fragmentation.LigandFragmentBlockMapping
   tmol.ligand.fragmentation.FragmentedLigandPoseMapping


Functions
---------

.. autoapisummary::

   tmol.ligand.fragmentation.recombine_fragmented_ligands
   tmol.ligand.fragmentation.fragment_ids_from_atom_array
   tmol.ligand.fragmentation.build_ligand_fragment_definition
   tmol.ligand.fragmentation.expand_fragmented_ligands
   tmol.ligand.fragmentation.apply_fragment_connections


Module Contents
---------------

.. py:data:: FRAGMENT_ID_ANNOTATION
   :value: 'tmol_fragment_id'


.. py:data:: MAX_FRAGMENT_CONNECTIONS
   :value: 4


.. py:data:: MIN_FRAGMENT_HEAVY_ATOMS
   :value: 3


.. py:class:: FragmentConnection

   .. rubric:: Docstring

   .. code-block:: text

      One directed side of a cut bond.
      

   .. py:attribute:: fragment_id
      :type:  int


   .. py:attribute:: partner_fragment_id
      :type:  int


   .. py:attribute:: connection_name
      :type:  str


   .. py:attribute:: partner_connection_name
      :type:  str


   .. py:attribute:: atom_name
      :type:  str


   .. py:attribute:: partner_atom_name
      :type:  str


   .. py:attribute:: bond_type
      :type:  str


.. py:class:: LigandFragmentDefinition

   .. rubric:: Docstring

   .. code-block:: text

      Structure-independent definition of one fragmented ligand type.
      

   .. py:attribute:: ligand_name
      :type:  str


   .. py:attribute:: atom_to_fragment
      :type:  Mapping[str, int]


   .. py:attribute:: fragment_ids
      :type:  tuple[int, Ellipsis]


   .. py:attribute:: fragment_preparations
      :type:  tuple[tmol.ligand.registry.LigandPreparation, Ellipsis]


   .. py:attribute:: connections
      :type:  tuple[FragmentConnection, Ellipsis]


   .. py:method:: fragment_name(fragment_id: int) -> str


.. py:class:: LigandFragmentBlockMapping

   .. rubric:: Docstring

   .. code-block:: text

      Map a user fragment ID onto its block in a built pose.
      

   .. py:attribute:: pose_index
      :type:  int


   .. py:attribute:: ligand_name
      :type:  str


   .. py:attribute:: residue_label
      :type:  int


   .. py:attribute:: pose_residue_label
      :type:  int


   .. py:attribute:: chain_label
      :type:  str


   .. py:attribute:: insertion_code
      :type:  str


   .. py:attribute:: fragment_id
      :type:  int


   .. py:attribute:: block_index
      :type:  int


   .. py:attribute:: atom_names
      :type:  tuple[str, Ellipsis]


   .. py:property:: fragment_name
      :type: str



.. py:class:: FragmentedLigandPoseMapping

   .. rubric:: Docstring

   .. code-block:: text

      Runtime mapping and connection list for a fragmented pose.
      

   .. py:attribute:: blocks
      :type:  tuple[LigandFragmentBlockMapping, Ellipsis]


   .. py:attribute:: connection_pairs
      :type:  tuple[tuple[int, str, int, str], Ellipsis]


   .. py:method:: split(pose_index: int) -> FragmentedLigandPoseMapping

      .. rubric:: Docstring

      .. code-block:: text

         Return the mapping for one pose, reindexed as pose zero.
         


.. py:function:: recombine_fragmented_ligands(structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack, mapping: FragmentedLigandPoseMapping) -> biotite.structure.AtomArray | biotite.structure.AtomArrayStack

   .. rubric:: Docstring

   .. code-block:: text

      Restore original residue identities on exported ligand fragments.
      

.. py:function:: fragment_ids_from_atom_array(atom_array: biotite.structure.AtomArray) -> numpy.ndarray | None

   .. rubric:: Docstring

   .. code-block:: text

      Return validated fragment IDs, or ``None`` when no split is requested.
      

.. py:function:: build_ligand_fragment_definition(preparation: tmol.ligand.registry.LigandPreparation, source_atom_array: biotite.structure.AtomArray) -> LigandFragmentDefinition | None

   .. rubric:: Docstring

   .. code-block:: text

      Partition a fully prepared ligand according to its source annotation.
      

.. py:function:: expand_fragmented_ligands(structure: biotite.structure.AtomArray | biotite.structure.AtomArrayStack, definitions: Sequence[LigandFragmentDefinition]) -> tuple[biotite.structure.AtomArray | biotite.structure.AtomArrayStack, FragmentedLigandPoseMapping]

   .. rubric:: Docstring

   .. code-block:: text

      Replace each annotated ligand residue with contiguous fragment residues.
      

.. py:function:: apply_fragment_connections(pose_stack, mapping: FragmentedLigandPoseMapping)

   .. rubric:: Docstring

   .. code-block:: text

      Install fragment cut bonds and rebuild all inter-block bond separations.
      

