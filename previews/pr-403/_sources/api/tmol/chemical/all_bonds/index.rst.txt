tmol.chemical.all_bonds
=======================

.. py:module:: tmol.chemical.all_bonds


Functions
---------

.. autoapisummary::

   tmol.chemical.all_bonds.bonds_and_bond_ranges


Module Contents
---------------

.. py:function:: bonds_and_bond_ranges(n_atoms: int, intra_res_bonds: tmol.types.array.NDArray[numpy.int32][:, 2], ordered_connection_atoms: tmol.types.array.NDArray[numpy.int32][:]) -> Tuple[tmol.types.array.NDArray[numpy.int32][:, 3], tmol.types.array.NDArray[numpy.int32][:, 2]]

   .. rubric:: Docstring

   .. code-block:: text

      Concatenate the set of intra- and inter-block bonds
      
      The intra-block bonds should list each bond twice, once for each of the
      two atoms it connects. The ordered-connection-atoms array lists the atom
      on this block that connects to another block such that the ith position
      in this array represents the ith inter-block connection; the cases when
      a single atom serves as a connection point to multiple other blocks
      (as might happen with metal ions) is handled correctly.
      
      This function returns a pair of arrays. The first array lists each bond
      as a tuple of (atom-ind, (other-atom)) where "other-atom" is similar to
      an unresolved-atom-id: it is itself a 2-tuple where if the indicated atom
      is a member of this block, then the first value in the tuple is a
      non-negative integer index of that atom, and if not, then the sentinel
      value of -1; if the atom is not a member of this block, then the second
      value in the tuple is the non-negative connection id, which can then
      be used to look up which other block, and which connection on that block
      is this block connected to in the PoseStack's data. The second array gives
      for each atom on this block the start and end indices of its bonds in
      the first array (all bonds in the first array for a single atom are
      contiguous).
      

