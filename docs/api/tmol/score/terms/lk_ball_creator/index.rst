tmol.score.terms.lk_ball_creator
================================

.. py:module:: tmol.score.terms.lk_ball_creator


Classes
-------

.. autoapisummary::

   tmol.score.terms.lk_ball_creator.LKBallTermCreator


Module Contents
---------------

.. py:class:: LKBallTermCreator

   Bases: :py:obj:`tmol.score.terms.term_creator.TermCreator`


   .. rubric:: Docstring

   .. code-block:: text

      Base class for registering score terms with the ScoreTermFactory.
      
      To add a new term,
      
        - add one or more new entries to the tmol.score.score_types enumeration
        - derive a new subclass of TermCreator and put it in this directory
          (the term itself should be implemented in a different directory)
        - the new TermCreator subclass needs to define two methods,
          create_term and score_types
        - create_term should instantiate the term
        - score_types should return a list of the elements of the score_types
          enumeration that the term implements in the order that the term
          will report them
      

   .. py:method:: create_term(param_db: tmol.database.ParameterDatabase, device: torch.device)
      :classmethod:



   .. py:method:: score_types()
      :classmethod:



