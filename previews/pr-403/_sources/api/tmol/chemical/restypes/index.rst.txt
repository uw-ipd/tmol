tmol.chemical.restypes
======================

.. py:module:: tmol.chemical.restypes


Attributes
----------

.. autoapisummary::

   tmol.chemical.restypes.AtomIndex
   tmol.chemical.restypes.ConnectionIndex
   tmol.chemical.restypes.BondCount
   tmol.chemical.restypes.BOND_TYPE_FROM_STR
   tmol.chemical.restypes.UnresolvedAtomID
   tmol.chemical.restypes.uaid_t
   tmol.chemical.restypes.ResName3
   tmol.chemical.restypes.IcoorIndex


Classes
-------

.. autoapisummary::

   tmol.chemical.restypes.BondType
   tmol.chemical.restypes.RefinedResidueType
   tmol.chemical.restypes.ResidueTypeSet


Functions
---------

.. autoapisummary::

   tmol.chemical.restypes.three2one
   tmol.chemical.restypes.one2three


Module Contents
---------------

.. py:data:: AtomIndex

.. py:data:: ConnectionIndex

.. py:data:: BondCount

.. py:class:: BondType

   Bases: :py:obj:`enum.IntEnum`


   .. rubric:: Docstring

   .. code-block:: text

      Enum where members are also (and must be) ints
      

   .. py:attribute:: SINGLE
      :value: 1



   .. py:attribute:: DOUBLE
      :value: 2



   .. py:attribute:: TRIPLE
      :value: 3



   .. py:attribute:: AROMATIC
      :value: 4



.. py:data:: BOND_TYPE_FROM_STR

.. py:data:: UnresolvedAtomID

.. py:data:: uaid_t

.. py:data:: ResName3

.. py:data:: IcoorIndex

.. py:function:: three2one(three: str) -> Union[str, None]

   .. rubric:: Docstring

   .. code-block:: text

      Return the one-letter amino acid code given its three letter code,
      or None if not a valid three-letter code
      

.. py:function:: one2three(one: str) -> Union[str, None]

   .. rubric:: Docstring

   .. code-block:: text

      Return the three-letter amino acid code given its one-letter code,
      or None if not a valid one-letter code.
      

.. py:class:: RefinedResidueType

   Bases: :py:obj:`tmol.database.chemical.RawResidueType`


   .. py:property:: n_atoms


   .. py:attribute:: atom_names_set
      :type:  Set[str]


   .. py:attribute:: atom_to_idx
      :type:  Mapping[str, AtomIndex]


   .. py:attribute:: aliases_map
      :type:  Mapping[str, str]


   .. py:attribute:: coord_dtype
      :type:  numpy.dtype


   .. py:attribute:: bond_indices
      :type:  numpy.ndarray


   .. py:attribute:: bond_to_type
      :type:  Mapping


   .. py:attribute:: bond_to_ringness
      :type:  Mapping


   .. py:property:: n_conn


   .. py:attribute:: connection_to_idx
      :type:  Mapping[str, AtomIndex]


   .. py:attribute:: connection_to_cidx
      :type:  Mapping[Optional[str], ConnectionIndex]


   .. py:attribute:: ordered_connection_atoms
      :type:  numpy.ndarray


   .. py:attribute:: connection_bond_types
      :type:  numpy.ndarray


   .. py:attribute:: all_bonds
      :type:  numpy.ndarray


   .. py:attribute:: down_connection_ind
      :type:  int


   .. py:attribute:: up_connection_ind
      :type:  int


   .. py:attribute:: torsion_to_uaids
      :type:  Mapping[str, Tuple[UnresolvedAtomID]]


   .. py:attribute:: ordered_torsions
      :type:  numpy.ndarray


   .. py:property:: n_torsions


   .. py:attribute:: is_torsion_mc
      :type:  numpy.ndarray


   .. py:attribute:: mc_torsions
      :type:  numpy.ndarray


   .. py:property:: n_mc_torsions


   .. py:attribute:: sc_torsions
      :type:  numpy.ndarray


   .. py:property:: n_sc_torsions


   .. py:attribute:: which_mcsc_torsion
      :type:  numpy.ndarray


   .. py:attribute:: path_distance
      :type:  numpy.ndarray


   .. py:attribute:: atom_paths_from_conn
      :type:  numpy.ndarray


   .. py:attribute:: atom_downstream_of_conn
      :type:  numpy.ndarray


   .. py:property:: n_icoors


   .. py:attribute:: icoors_index
      :type:  Mapping[str, IcoorIndex]


   .. py:attribute:: at_to_icoor_ind
      :type:  numpy.ndarray


   .. py:attribute:: icoors_ancestors
      :type:  numpy.ndarray


   .. py:attribute:: icoors_geom
      :type:  numpy.ndarray


   .. py:attribute:: ideal_coords
      :type:  numpy.ndarray


   .. py:method:: compute_ideal_coords()


   .. py:attribute:: default_jump_connection_atom_index
      :type:  int


   .. py:method:: get_default_jump_connection_atom_index()


.. py:class:: ResidueTypeSet

   .. py:method:: get_default() -> ResidueTypeSet
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Load and return the residue type set constructed from the default param db
         


   .. py:method:: from_database(chemical_db: tmol.chemical.patched_chemdb.PatchedChemicalDatabase)
      :classmethod:



   .. py:method:: from_restype_list(chemical_db: tmol.chemical.patched_chemdb.PatchedChemicalDatabase, restypes: List[RefinedResidueType])
      :classmethod:



   .. py:attribute:: residue_types
      :type:  Sequence[RefinedResidueType]


   .. py:attribute:: restype_map
      :type:  Mapping[ResName3, Sequence[RefinedResidueType]]


   .. py:attribute:: chem_db
      :type:  tmol.chemical.patched_chemdb.PatchedChemicalDatabase


