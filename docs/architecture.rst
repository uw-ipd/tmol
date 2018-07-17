.. _architecture:

=================
Architecture
=================

TMol is roughly divided into two core modules: a high-level description of
polymeric molecules in `tmol.system` and a low-level, term-specific description
used `tmol.score`. 

`tmol.system` and `tmol.scoring` are coupled via
:py:func:`~functools.singledispatch` hooks registered in
`tmol.system.score_support`.

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

