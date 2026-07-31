tmol.io.canonical_form
======================

.. py:module:: tmol.io.canonical_form


Classes
-------

.. autoapisummary::

   tmol.io.canonical_form.CanonicalForm


Module Contents
---------------

.. py:class:: CanonicalForm

   .. rubric:: Docstring

   .. code-block:: text

      This class holds the data that describe a (stack of) structure(s) in a poised, ready-to-use state.
      
      This datastructure holds the information necessary to determine the chemical
      identities of the residues in the structure(s), which may be under-determined
      from tmol's perspective by the source of the structure (e.g. OpenFold does not
      explicitly model termini). The atoms that are present are represented with
      non-NaN coordinates in the `coords` array; the order in which those atoms appear
      is given by a particular CanonicalOrdering object.
      
      The datastructure also holds convenience information such as author-provided
      residue labels (ints), chain labels (strings) & insertion codes (strings) as well
      as the occupancy and B-factor of each atom. These are not strictly necessary
      but are often useful when processing structures.
      

   .. py:attribute:: chain_id
      :type:  tmol.types.torch.Tensor[torch.int64][:, :]


   .. py:attribute:: res_types
      :type:  tmol.types.torch.Tensor[torch.int64][:, :]


   .. py:attribute:: coords
      :type:  tmol.types.torch.Tensor[torch.float32][:, :, :, 3]


   .. py:attribute:: res_labels
      :type:  tmol.types.array.NDArray[int][:, :]


   .. py:attribute:: residue_insertion_codes
      :type:  tmol.types.array.NDArray[object][:, :]


   .. py:attribute:: chain_labels
      :type:  tmol.types.array.NDArray[object][:, :]


   .. py:attribute:: atom_occupancy
      :type:  Optional[tmol.types.array.NDArray[numpy.float32][:, :, :]]


   .. py:attribute:: atom_b_factor
      :type:  Optional[tmol.types.array.NDArray[numpy.float32][:, :, :]]


   .. py:attribute:: disulfides
      :type:  Optional[tmol.types.torch.Tensor[torch.int64][:, 3]]


   .. py:attribute:: res_not_connected
      :type:  Optional[tmol.types.torch.Tensor[torch.bool][:, :, 2]]


   .. py:method:: as_dict()


