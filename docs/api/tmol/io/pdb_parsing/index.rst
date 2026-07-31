tmol.io.pdb_parsing
===================

.. py:module:: tmol.io.pdb_parsing

.. rubric:: Module docstring

.. code-block:: text

   Utility functions for converting pdb files to/from atom records.
   
   Atom records are DataFrames with the records:
   
       "model"       : Integer model number
       "model_name"  : String model name
       "record_name" : PDB record name, assumed to be 'ATOM'
       "atomi"       : (Below) As per PDB spec, whitespace trimmed strings and parsed numeric values
       "atomn"
       "location"
       "resn"
       "chain"
       "resi"
       "insert"
       "x"
       "y"
       "z"
       "occupancy"
       "b"
   


Attributes
----------

.. autoapisummary::

   tmol.io.pdb_parsing.atom_record_dtype


Functions
---------

.. autoapisummary::

   tmol.io.pdb_parsing.parse_pdb
   tmol.io.pdb_parsing.parse_atom_lines
   tmol.io.pdb_parsing.to_pdb
   tmol.io.pdb_parsing.format_atomn
   tmol.io.pdb_parsing.to_pdb_lines
   tmol.io.pdb_parsing.to_atom_lines


Module Contents
---------------

.. py:data:: atom_record_dtype

.. py:function:: parse_pdb(pdb_lines) -> pandas.DataFrame

   .. rubric:: Docstring

   .. code-block:: text

      Parses pdb file into atom records.
      
      pdb_lines : Iterable lines, a string filename, or a string of lines in PDB format.
      

.. py:function:: parse_atom_lines(lines)

   .. rubric:: Docstring

   .. code-block:: text

      Parses an array of pdb ATOM records into a dict of field arrays.
      
      1 -  6         Record name     "ATOM  "
      7 - 11         Integer         Atom serial number.
      13 - 16        Atom            Atom name.
      17             Character       Alternate location indicator.
      18 - 20        Residue name    Residue name.
      22             Character       Chain identifier.
      23 - 26        Integer         Residue sequence number.
      27             AChar           Code for insertion of residues.
      31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
      39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
      47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.
      55 - 60        Real(6.2)       Occupancy.
      61 - 66        Real(6.2)       Temperature factor (Default = 0.0).
      73 - 76        LString(4)      Segment identifier, left-justified.
      77 - 78        LString(2)      Element symbol, right-justified.
      79 - 80        LString(2)      Charge on the atom.
      

.. py:function:: to_pdb(atom_records)

   .. rubric:: Docstring

   .. code-block:: text

      Atom record DataFrame as pdb text.
      

.. py:function:: format_atomn(atomn)

   .. rubric:: Docstring

   .. code-block:: text

      Formats atomn via pdb standard.
      
      If atomn is a single-letter element (N, C, O, S, H), then printed atomn record of the the format ' {atomn:<3}', else of the format '{atomn:<4}'
      

.. py:function:: to_pdb_lines(atom_records)

   .. rubric:: Docstring

   .. code-block:: text

      Yields atom record DataFrame as pdb lines.
      

.. py:function:: to_atom_lines(atom_records)

   .. rubric:: Docstring

   .. code-block:: text

      Convert atom records into ATOM lines.
      

