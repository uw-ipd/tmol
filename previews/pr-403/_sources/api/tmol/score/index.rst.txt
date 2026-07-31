tmol.score
==========

.. py:module:: tmol.score


Submodules
----------

.. toctree::
   :maxdepth: 1

   /api/tmol/score/atom_type_dependent_term/index
   /api/tmol/score/backbone_torsion/index
   /api/tmol/score/bond_dependent_term/index
   /api/tmol/score/bonded_atom/index
   /api/tmol/score/cartbonded/index
   /api/tmol/score/chemical_database/index
   /api/tmol/score/common/index
   /api/tmol/score/disulfide/index
   /api/tmol/score/dunbrack/index
   /api/tmol/score/elec/index
   /api/tmol/score/energy_term/index
   /api/tmol/score/genbonded/index
   /api/tmol/score/hbond/index
   /api/tmol/score/ljlk/index
   /api/tmol/score/lk_ball/index
   /api/tmol/score/ref/index
   /api/tmol/score/score_function/index
   /api/tmol/score/score_types/index
   /api/tmol/score/score_utils/index
   /api/tmol/score/terms/index


Functions
---------

.. autoapisummary::

   tmol.score.beta2016_score_function


Package Contents
----------------

.. py:function:: beta2016_score_function(device: torch.device, param_db: Optional[tmol.database.ParameterDatabase] = None) -> score_function.ScoreFunction

   .. rubric:: Docstring

   .. code-block:: text

      Return a ScoreFunction implementing the beta_nov2016 score function
      of Rosetta3.
      
      :param device: Target torch device.
      :param param_db: Optional parameter database. If omitted, uses the process
                       default parameter database and a memoized score function.
      
      :returns: Configured `ScoreFunction`.
      
      When `param_db` is provided, this creates a fresh score function
      (no memoization — caller owns database lifecycle).
      
      See:
      https://pubs.acs.org/doi/10.1021/acs.jctc.6b0081 and
      https://pubs.acs.org/doi/full/10.1021/acs.jctc.7b00125
      

