.. _architecture:

============
Architecture
============

TMol is roughly divided into two core modules: a high-level description of
polymeric molecules in `tmol.system` and a low-level, term-specific description
used `tmol.score`.


These components operate over a shared chemical vocabulary defined in
`tmol.database.chemical`, with additional term-specific data given in
`tmol.database.scoring`.

.. aafig::

  +--------+          +---------+
  |        |          |         |
  | system +----------o scoring |
  |        |          |         |
  +--+-----+          +--+----+-+
     |                   |    |
     | +-----------------v-+  |
     | |                   |  |
     | | database.scoring  |  |
     | |                   |  |
     | +---------+---------+  |
     |           |            |
     | +---------v---------+  |
     | |                   |  |
     +-> database.chemical <--+
       |                   |
       +-------------------+


`tmol.system` and `tmol.score` are coupled via
:py:func:`~functools.singledispatch` hooks registered in
`tmol.system.score_support`.

Scoring
=======

* Scoring is executed via a dynamic "score graph" component, managing
creation of a torch compute graph for the current score model.
* Score model is partitioned into score graph component classes, each
covering a logically distinct component of the score model. This may be
a score term, or support data required for score evaluation.
* Component classes are combined as mixins into a reactive score graph,
reference reactive docs.
* Initialization is handled via cooperative factory. 

Scoring model operates on system stacks, a batch of systems of equivalent
maximum size. Score operations are defined for "intra-layer" scores,
mapping a [L, N] stack into [L] intra-stack scores, or "inter-layer" scores, mapping 
a [L1, N1], [L2, N2] pair of stacks into an [L1, L2] inter-stack score
matrix.
