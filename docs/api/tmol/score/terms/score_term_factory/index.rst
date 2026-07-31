tmol.score.terms.score_term_factory
===================================

.. py:module:: tmol.score.terms.score_term_factory


Classes
-------

.. autoapisummary::

   tmol.score.terms.score_term_factory.ScoreTermFactory


Module Contents
---------------

.. py:class:: ScoreTermFactory

   .. rubric:: Docstring

   .. code-block:: text

      Factory for the creation of EnergyTerms
      
      This class uses import-time factory registration to discover the set
      of TermCreators that live in the same directory as it. To register
      a new TermCreator, simply put the term creator in this directory.
      

   .. py:attribute:: creator_map
      :type:  Dict


   .. py:method:: factory_register(creator: TermCreator)
      :classmethod:



   .. py:method:: create_term_for_score_type(st: tmol.score.score_types.ScoreType, param_db: tmol.database.ParameterDatabase, device: torch.device)
      :classmethod:



