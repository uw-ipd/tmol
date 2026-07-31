tmol.database.scoring.genbonded
===============================

.. py:module:: tmol.database.scoring.genbonded


Classes
-------

.. autoapisummary::

   tmol.database.scoring.genbonded.GenBondedTorsionEntry
   tmol.database.scoring.genbonded.GenBondedImproperEntry
   tmol.database.scoring.genbonded.GenBondedDatabase


Module Contents
---------------

.. py:class:: GenBondedTorsionEntry

   .. rubric:: Docstring

   .. code-block:: text

      One torsion entry from the genbonded parameter database.
      
      atoms  -- four atom-type strings (may be generic, e.g. "C*", or wildcard "X")
      bond   -- bond character:
                  ~ any bond
                  @ ring bond
                  - single bond
                  = double bond
                  # triple bond
                  : aromatic bond
      k1..k4 -- Fourier coefficients for periods 1-4
      offset -- single phase offset applied to all Fourier terms
      

   .. py:attribute:: atoms
      :type:  Tuple[str, str, str, str]


   .. py:attribute:: bond
      :type:  str


   .. py:attribute:: k1
      :type:  float
      :value: 0.0



   .. py:attribute:: k2
      :type:  float
      :value: 0.0



   .. py:attribute:: k3
      :type:  float
      :value: 0.0



   .. py:attribute:: k4
      :type:  float
      :value: 0.0



   .. py:attribute:: offset
      :type:  float
      :value: 0.0



.. py:class:: GenBondedImproperEntry

   .. rubric:: Docstring

   .. code-block:: text

      One improper torsion entry from the genbonded parameter database.
      
      atoms  -- four atom-type strings: atoms[0] = center, atoms[1..3] = bonded
             -- (may be generic, e.g. "C*", or wildcard "X")
      k      -- harmonic spring constant  (E = k*(theta - delta)^2)
      delta  -- ideal improper torsion angle (radians)
      

   .. py:attribute:: atoms
      :type:  Tuple[str, str, str, str]


   .. py:attribute:: k
      :type:  float
      :value: 0.0



   .. py:attribute:: delta
      :type:  float
      :value: 0.0



.. py:class:: GenBondedDatabase

   .. rubric:: Docstring

   .. code-block:: text

      Database for the genbonded (generic-bonded torsional) scoring term.
      
      atom_hierarchy -- maps each concrete atom-type name to an ordered list of
                        types to try when looking up a parameter entry, from most
                        specific to most generic.  e.g. {"CS": ["CS", "C*", "X"]}
      
      torsions       -- ordered list of torsion entries (most-specific first so
                        that a linear scan finds the best match quickly).
      
      impropers      -- ordered list of improper torsion entries (center + 3
                        bonded atoms; order of bonded atoms is unordered).
      
      coverage       -- maps each atom-type string to the count of concrete types
                        whose hierarchy includes that string.  Used for Rosetta-
                        style multiplicity scoring.
      
      multi_max      -- half the total number of concrete atom types; scales the
                        bond-type contribution to multiplicity so that bond
                        specificity dominates atom-type specificity (mirrors
                        Rosetta's multi_max = indicesX.size() / 2).
      

   .. py:attribute:: atom_hierarchy
      :type:  Dict[str, List[str]]


   .. py:attribute:: torsions
      :type:  Tuple[GenBondedTorsionEntry, Ellipsis]


   .. py:attribute:: impropers
      :type:  Tuple[GenBondedImproperEntry, Ellipsis]


   .. py:attribute:: coverage
      :type:  Dict[str, int]


   .. py:attribute:: multi_max
      :type:  int


   .. py:method:: from_file(path: str) -> GenBondedDatabase
      :classmethod:



   .. py:method:: all_type_names() -> List[str]

      .. rubric:: Docstring

      .. code-block:: text

         Sorted list of all unique chemical type strings in the database.
         


   .. py:method:: make_type_to_idx() -> Dict[str, int]

      .. rubric:: Docstring

      .. code-block:: text

         Return a dict mapping every chemical type string to a unique int index.
         


   .. py:method:: hierarchy_for(atom_type: str) -> List[str]

      .. rubric:: Docstring

      .. code-block:: text

         Return the fallback list for *atom_type*, defaulting to [atom_type].
         


   .. py:method:: find_torsion_params(type1: str, type2: str, type3: str, type4: str, bond_type_int: int, is_ring: bool) -> Optional[GenBondedTorsionEntry]

      .. rubric:: Docstring

      .. code-block:: text

         Return the best-matching torsion entry using Rosetta's multiplicity scoring.
         
         Multiplicity = bond_bin_count * multi_max^4 + coverage(a1)*...*coverage(a4)
         
         Bond specificity dominates: an entry with a more-specific bond char
         (fewer covered bins) beats one with a less-specific bond char regardless
         of atom-type generality, as long as both match.  Within the same bond
         specificity, atom-type coverage breaks ties (lower = more specific).
         
         This mirrors Rosetta's GenericBondedPotential multiplicity formula exactly:
           multBT   = indicesBT.size() * multi_max^4
           mult_atm = indices1.size() * indices2.size() * indices3.size() * indices4.size()
           multiplicity = multBT + mult_atm   (lower = more specific = preferred)
         


   .. py:method:: find_improper_params(center: str, n1: str, n2: str, n3: str) -> Optional[GenBondedImproperEntry]

      .. rubric:: Docstring

      .. code-block:: text

         Return the best-matching improper entry for (center, n1, n2, n3).
         
         The three bonded atoms (n1, n2, n3) are considered unordered: all six
         permutations are tried.  'Best' is the lowest total hierarchy-position
         score.  Returns None if no match is found.
         


