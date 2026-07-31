tmol.score.chemical_database
============================

.. py:module:: tmol.score.chemical_database


Classes
-------

.. autoapisummary::

   tmol.score.chemical_database.AcceptorHybridization
   tmol.score.chemical_database.AtomTypeParams
   tmol.score.chemical_database.AtomTypeParamResolver


Module Contents
---------------

.. py:class:: AcceptorHybridization

   Bases: :py:obj:`enum.IntEnum`


   .. rubric:: Docstring

   .. code-block:: text

      Enum where members are also (and must be) ints
      

   .. py:attribute:: none
      :value: 0



   .. py:attribute:: sp2
      :value: 1



   .. py:attribute:: sp3
      :value: 2



   .. py:attribute:: ring
      :value: 3



.. py:class:: AtomTypeParams

   Bases: :py:obj:`tmol.types.tensor.TensorGroup`, :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. py:attribute:: is_acceptor
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


   .. py:attribute:: acceptor_hybridization
      :type:  tmol.types.torch.Tensor[torch.int32][Ellipsis]


   .. py:attribute:: is_donor
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


   .. py:attribute:: is_hydrogen
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


   .. py:attribute:: is_hydroxyl
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


   .. py:attribute:: is_polarh
      :type:  tmol.types.torch.Tensor[bool][Ellipsis]


.. py:class:: AtomTypeParamResolver

   Bases: :py:obj:`tmol.types.attrs.ValidateAttrs`


   .. rubric:: Docstring

   .. code-block:: text

      Container for global/type/pair parameters, indexed by atom type name.
      
      Param resolver stores pair parameters for a collection of atom types, using
      a pandas Index to map from string atom type to a resolver-specific integer type
      index.
      

   .. py:attribute:: index
      :type:  pandas.Index


   .. py:attribute:: params
      :type:  AtomTypeParams


   .. py:attribute:: device
      :type:  torch.device


   .. py:method:: type_idx(atom_types: tmol.types.array.NDArray[object][Ellipsis]) -> tmol.types.torch.Tensor[torch.int64][Ellipsis]

      .. rubric:: Docstring

      .. code-block:: text

         Convert array of atom type names to parameter indices.
         
         pandas.Index.get_indexer only operates on 1-d input arrays. Coerces
         higher-dimensional arrays, as may be produced via broadcasting, into
         lower-dimensional views to resolver parameter indices.
         


   .. py:method:: from_database(chemical_database: tmol.database.chemical.ChemicalDatabase, device: torch.device)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Initialize param resolver for all atom types in database.
         


