tmol.score.bonded_atom
======================

.. py:module:: tmol.score.bonded_atom


Classes
-------

.. autoapisummary::

   tmol.score.bonded_atom.IndexedBonds


Module Contents
---------------

.. py:class:: IndexedBonds

   .. py:attribute:: bonds
      :type:  tmol.types.torch.Tensor[int][:, :, 2]


   .. py:attribute:: bond_spans
      :type:  tmol.types.torch.Tensor[int][:, :, 2]


   .. py:method:: from_bonds(src_bonds, minlength=None)
      :classmethod:



   .. py:method:: to_directed(src_bonds)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Convert a potentially-undirected bond-table into dense, directed bonds.
         The input "bonds" tensor is a two dimensional array of nbonds x 3,
         where the 2nd dimension holds [stack index, atom 1 index, atom 2 index].
         
         Eg. Converts
         [[0, 0, 1], [0, 0, 2]]
         into
         [[0, 0, 1], [0, 1, 0], [0, 0, 2], [0, 2, 0]]
         


