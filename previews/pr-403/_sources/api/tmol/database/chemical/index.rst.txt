tmol.database.chemical
======================

.. py:module:: tmol.database.chemical


Attributes
----------

.. autoapisummary::

   tmol.database.chemical.AcceptorHybridization


Classes
-------

.. autoapisummary::

   tmol.database.chemical.Element
   tmol.database.chemical.AtomType
   tmol.database.chemical.Atom
   tmol.database.chemical.AtomAlias
   tmol.database.chemical.Icoor
   tmol.database.chemical.Connection
   tmol.database.chemical.UnresolvedAtom
   tmol.database.chemical.Torsion
   tmol.database.chemical.ChiSamples
   tmol.database.chemical.SidechainBuilding
   tmol.database.chemical.PolymerProperties
   tmol.database.chemical.ProtonationProperties
   tmol.database.chemical.ChemicalProperties
   tmol.database.chemical.RawResidueType
   tmol.database.chemical.IcoorVariant
   tmol.database.chemical.PolymerPropertiesVariant
   tmol.database.chemical.ChemicalPropertiesVariant
   tmol.database.chemical.VariantType
   tmol.database.chemical.ChemicalDatabase


Functions
---------

.. autoapisummary::

   tmol.database.chemical.normalize_bond_tuples


Package Contents
----------------

.. py:data:: AcceptorHybridization

.. py:function:: normalize_bond_tuples(raw)

   .. rubric:: Docstring

   .. code-block:: text

      Normalize legacy 2-field bond entries to include bond order.
      
      Historically, some YAML snippets used ``[atom1, atom2]`` for bonds.
      The typed schema expects 3-tuples: ``(atom1, atom2, bond_type)``.
      This helper expands 2-field entries to use ``"SINGLE"`` as default.
      
      Handles both the top-level dict shape (``chemical.yaml``) and a
      flat list of residue/variant dicts.
      

.. py:class:: Element

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: atomic_number
      :type:  int


.. py:class:: AtomType

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: element
      :type:  str


   .. py:attribute:: is_acceptor
      :type:  bool
      :value: False



   .. py:attribute:: is_donor
      :type:  bool
      :value: False



   .. py:attribute:: is_hydroxyl
      :type:  bool
      :value: False



   .. py:attribute:: is_polarh
      :type:  bool
      :value: False



   .. py:attribute:: acceptor_hybridization
      :type:  Optional[AcceptorHybridization]
      :value: None



.. py:class:: Atom

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: atom_type
      :type:  str


.. py:class:: AtomAlias

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: alt_name
      :type:  str


.. py:class:: Icoor

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: phi
      :type:  tmol.utility.units.DihedralAngle


   .. py:attribute:: theta
      :type:  tmol.utility.units.BondAngle


   .. py:attribute:: d
      :type:  float


   .. py:attribute:: parent
      :type:  str


   .. py:attribute:: grand_parent
      :type:  str


   .. py:attribute:: great_grand_parent
      :type:  str


.. py:class:: Connection

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: atom
      :type:  str


   .. py:attribute:: type
      :type:  str
      :value: 'SINGLE'



.. py:class:: UnresolvedAtom

   .. py:attribute:: atom
      :type:  Optional[str]
      :value: None



   .. py:attribute:: connection
      :type:  Optional[str]
      :value: None



   .. py:attribute:: bond_sep_from_conn
      :type:  Optional[int]
      :value: None



.. py:class:: Torsion

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: a
      :type:  UnresolvedAtom


   .. py:attribute:: b
      :type:  UnresolvedAtom


   .. py:attribute:: c
      :type:  UnresolvedAtom


   .. py:attribute:: d
      :type:  UnresolvedAtom


.. py:class:: ChiSamples

   .. py:attribute:: chi_dihedral
      :type:  str


   .. py:attribute:: samples
      :type:  Tuple[float, Ellipsis]


   .. py:attribute:: expansions
      :type:  Tuple[float, Ellipsis]


.. py:class:: SidechainBuilding

   .. py:attribute:: chi_samples
      :type:  ChiSamples


.. py:class:: PolymerProperties

   .. py:attribute:: is_polymer
      :type:  bool


   .. py:attribute:: polymer_type
      :type:  str


   .. py:attribute:: backbone_type
      :type:  str


   .. py:attribute:: mainchain_atoms
      :type:  Optional[Tuple[str, Ellipsis]]


   .. py:attribute:: sidechain_chirality
      :type:  str


   .. py:attribute:: termini_variants
      :type:  Tuple[str, Ellipsis]


.. py:class:: ProtonationProperties

   .. py:attribute:: protonated_atoms
      :type:  Tuple[str, Ellipsis]


   .. py:attribute:: protonation_state
      :type:  str


   .. py:attribute:: pH
      :type:  float


.. py:class:: ChemicalProperties

   .. py:attribute:: is_canonical
      :type:  bool


   .. py:attribute:: polymer
      :type:  PolymerProperties


   .. py:attribute:: chemical_modifications
      :type:  Tuple[str, Ellipsis]


   .. py:attribute:: connectivity
      :type:  Tuple[str, Ellipsis]


   .. py:attribute:: protonation
      :type:  ProtonationProperties


   .. py:attribute:: virtual
      :type:  Tuple[str, Ellipsis]


.. py:class:: RawResidueType

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: base_name
      :type:  str


   .. py:attribute:: name3
      :type:  str


   .. py:attribute:: io_equiv_class
      :type:  str


   .. py:attribute:: atoms
      :type:  Tuple[Atom, Ellipsis]


   .. py:attribute:: atom_aliases
      :type:  Tuple[AtomAlias, Ellipsis]


   .. py:attribute:: bonds
      :type:  Tuple[tuple, Ellipsis]


   .. py:attribute:: connections
      :type:  Tuple[Connection, Ellipsis]


   .. py:attribute:: torsions
      :type:  Tuple[Torsion, Ellipsis]


   .. py:attribute:: icoors
      :type:  Tuple[Icoor, Ellipsis]


   .. py:attribute:: properties
      :type:  ChemicalProperties


   .. py:attribute:: chi_samples
      :type:  Tuple[ChiSamples, Ellipsis]


   .. py:attribute:: default_jump_connection_atom
      :type:  str


   .. py:attribute:: hydrogens_regenerated
      :type:  bool
      :value: False



   .. py:attribute:: is_ligand_fragment
      :type:  bool
      :value: False



   .. py:method:: atom_name(index)


.. py:class:: IcoorVariant

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: source
      :type:  Optional[str]
      :value: None



   .. py:attribute:: phi
      :type:  Optional[tmol.utility.units.DihedralAngle]
      :value: 0.0



   .. py:attribute:: theta
      :type:  Optional[tmol.utility.units.BondAngle]
      :value: 0.0



   .. py:attribute:: d
      :type:  Optional[float]
      :value: 0.0



   .. py:attribute:: parent
      :type:  Optional[str]
      :value: None



   .. py:attribute:: grand_parent
      :type:  Optional[str]
      :value: None



   .. py:attribute:: great_grand_parent
      :type:  Optional[str]
      :value: None



.. py:class:: PolymerPropertiesVariant

   .. py:attribute:: polymer_type
      :type:  str


.. py:class:: ChemicalPropertiesVariant

   .. py:attribute:: polymer
      :type:  Optional[PolymerPropertiesVariant]
      :value: None



.. py:class:: VariantType

   .. py:attribute:: name
      :type:  str


   .. py:attribute:: display_name
      :type:  str


   .. py:attribute:: pattern
      :type:  str


   .. py:attribute:: remove_atoms
      :type:  Tuple[str, Ellipsis]


   .. py:attribute:: add_atoms
      :type:  Tuple[Atom, Ellipsis]


   .. py:attribute:: add_atom_aliases
      :type:  Tuple[AtomAlias, Ellipsis]


   .. py:attribute:: modify_atoms
      :type:  Tuple[Atom, Ellipsis]


   .. py:attribute:: add_connections
      :type:  Tuple[Connection, Ellipsis]


   .. py:attribute:: add_bonds
      :type:  Tuple[tuple, Ellipsis]


   .. py:attribute:: icoors
      :type:  Tuple[IcoorVariant, Ellipsis]


.. py:class:: ChemicalDatabase

   .. py:attribute:: element_types
      :type:  Tuple[Element, Ellipsis]


   .. py:attribute:: atom_types
      :type:  Tuple[AtomType, Ellipsis]


   .. py:attribute:: residues
      :type:  Tuple[RawResidueType, Ellipsis]


   .. py:attribute:: variants
      :type:  Tuple[VariantType, Ellipsis]


   .. py:method:: get_default() -> ChemicalDatabase
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Load and return default parameter database.
         


   .. py:method:: from_file(path)
      :classmethod:



