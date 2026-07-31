tmol.score.score_function
=========================

.. py:module:: tmol.score.score_function


Attributes
----------

.. autoapisummary::

   tmol.score.score_function.logger
   tmol.score.score_function.SFXN_FORMAT_VERSION


Classes
-------

.. autoapisummary::

   tmol.score.score_function.ScoreFunction
   tmol.score.score_function.WholePoseScoringModule
   tmol.score.score_function.BlockPairScoringModule
   tmol.score.score_function.RotamerScoringModule


Module Contents
---------------

.. py:data:: logger

.. py:data:: SFXN_FORMAT_VERSION
   :type:  str
   :value: '1.0'


.. py:class:: ScoreFunction(param_db: tmol.database.ParameterDatabase, device: torch.device)

   .. py:attribute:: term_options


   .. py:method:: set_weight(st: tmol.score.score_types.ScoreType, weight: float)


   .. py:method:: get_weight(st: tmol.score.score_types.ScoreType)


   .. py:method:: score_type_covered_by_contained_term(st: tmol.score.score_types.ScoreType)


   .. py:method:: retrieve_term_for_score_type(st: tmol.score.score_types.ScoreType)


   .. py:method:: term_for_st_has_no_other_non_zero_weights(st: tmol.score.score_types.ScoreType)


   .. py:method:: all_terms()

      .. rubric:: Docstring

      .. code-block:: text

         Grant read access to the list of terms.
         
         Do not modify this list directly
         


   .. py:method:: all_score_types()


   .. py:method:: one_body_terms()


   .. py:method:: two_body_terms()


   .. py:method:: multi_body_terms()


   .. py:method:: render_whole_pose_scoring_module(pose_stack: tmol.pose.pose_stack.PoseStack)

      .. rubric:: Docstring

      .. code-block:: text

         Create an object designed to evaluate the score of a set of Poses
         repeatedly as the Poses change their conformation, e.g., as in
         minimization. This object will derive from torch.nn.Module and
         it will contain a set of objects rendered by the ScoreFunction's
         terms that themselves are derived from torch.nn.Module. This
         object's __call__ will return a tensor of weighted energies of
         shape (n_poses,).
         


   .. py:method:: render_block_pair_scoring_module(pose_stack: tmol.pose.pose_stack.PoseStack)

      .. rubric:: Docstring

      .. code-block:: text

         Create an object designed to evaluate the score of a set of Poses
         repeatedly as the Poses change their conformation, e.g., as in
         minimization. This object will derive from torch.nn.Module and
         it will contain a set of objects rendered by the ScoreFunction's
         terms that themselves are derived from torch.nn.Module. This
         object's __call__ will return a tensor of weighted energies of
         shape (n_poses, max_n_blocks, max_n_blocks).
         


   .. py:method:: render_rotamer_scoring_module(pose_stack: tmol.pose.pose_stack.PoseStack, rotamer_set: RotamerSet)

      .. rubric:: Docstring

      .. code-block:: text

         Create an object designed to evaluate the score a RotamerSet
         repeatedly as the Poses change their conformation, e.g., as in
         minimization. This object will derive from torch.nn.Module and
         it will contain a set of objects rendered by the ScoreFunction's
         terms that themselves are derived from torch.nn.Module. This
         object's __call__ will return a tensor of weighted energies of
         shape (n_poses, max_n_blocks, max_n_blocks).
         


   .. py:method:: pre_work_initialization(pose_stack: tmol.pose.pose_stack.PoseStack)


   .. py:method:: set_option(key: str, value)

      .. rubric:: Docstring

      .. code-block:: text

         Set an option for all energy terms.
         
         Options are passed to each energy term's set_options method
         as a dictionary during pre_work_initialization.
         


   .. py:method:: set_options(options: Dict)

      .. rubric:: Docstring

      .. code-block:: text

         Set the score function options by a dict.
         
         This replaces the options dict entirely - any previous values
         are gone.
         


   .. py:method:: weights_tensor()


   .. py:method:: from_sfxn_file(path, param_db, device)
      :classmethod:


      .. rubric:: Docstring

      .. code-block:: text

         Create a ScoreFunction from a YAML weights file.
         
         :param path: Path to a YAML file containing a ``weights`` dict mapping
                      score type names (as in ``ScoreType``) to their weights, as well
                      as any other options to configure the score function.
         :param param_db: ParameterDatabase instance.
         :param device: Target torch device.
         
         :returns: Configured ScoreFunction with all weights from the file applied.
         


   .. py:method:: get_sorted_terms(term_list)
      :staticmethod:



.. py:class:: WholePoseScoringModule(weights: tmol.types.torch.Tensor[torch.float32][:], term_modules: Sequence[torch.nn.Module])

   .. py:attribute:: weights


   .. py:attribute:: term_modules


   .. py:method:: unweighted_scores(coords)


.. py:class:: BlockPairScoringModule(weights: tmol.types.torch.Tensor[torch.float32][:], term_modules: Sequence[torch.nn.Module])

   .. py:attribute:: weights


   .. py:attribute:: term_modules


   .. py:method:: unweighted_scores(coords)


.. py:class:: RotamerScoringModule(weights: tmol.types.torch.Tensor[torch.float32][:], term_modules: Sequence[torch.nn.Module])

   .. py:attribute:: weights


   .. py:attribute:: term_modules


