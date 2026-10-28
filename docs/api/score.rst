Scoring
-------

Use :func:`tmol.beta2016_score_function` for the standard configured score
function, or construct :class:`tmol.ScoreFunction` and set individual
:class:`tmol.ScoreType` weights. These, and
:func:`tmol.calculate_fragment_interactions`, are imported from the top-level
package and documented in :doc:`top_level`. The energy-term classes,
constraint helpers, and :class:`tmol.score.ScoreTermFactory` exported by
:mod:`tmol.score` are documented in :doc:`score_terms`.

.. automodule:: tmol.score
