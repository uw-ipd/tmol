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

Scoring is managed via a "score graph" object, managing the initialization
of a torch compute graph calculating a setup of score terms for
a collection model states.

A model is defined over of a set ``n`` of bonded atoms. Each atom is located at
an atom index, and is defined by a type and coordinate. Atoms may be "null",
defining no type and a nan coordinate at a given index.  Bonds are defined via
a set of ``b``` sparse, undirected bonded inter-atom index pairs.

A score graph operates over a "depth" of ``l`` layers, each containing a single
model. Models must contain the same number of atoms ``n``, but may have
differing atom types and null atoms. Bonds are strictly intra-layer, and form
a disjoint set per-layer inter-atom graphs.


.. aafig::

  +---------------------------------------+
  |                                  --   |
  | "[n] atom_types"                /  \  |
  | "[n] coordinates"            +-+    + |
  | "[b] (a,b) bond indices"    /   \  /  |
  |                                  --   |
  +----------------------------------+----+
                                     |
                                     |
  +----------------------------------|----+
  |                              +---o--+ |
  |                              +------+ |
  | "[l] layers"                 +------+ |
  |                              +------+ |
  |                              +------+ |
  +---------------------------------------+

Score calculation is performed on an intra-layer and inter-layer basis.
Intra-layer scoring is defined over across all interactions (bonded and
non-bonded) within a layer, yielding ``l`` scores for a single score graph of
depth ``l``. Inter-layer scoring is defined over all inter-layer non-bonded
interactions, yielding a ``[i, j]`` pairwise score array for two score graphs
of depth ``i`` and ``j``.

.. note:: `tmol.score` currently only supports intra-layer scoring, and is
   limited to models of depth 1.

The score graph implementation is partitioned into score component classes,
each covering a logically distinct component of the score function. These
components include score terms, derived model representations, or support data
required for score evaluation. Component classes are combined as mixins into
a `reactive` score graph. At minimum, a valid score graph will include an
atomic representation, some number of score terms, and a total score property.

.. aafig::

  +------------------------------+
  |                              |
  |          +-------+           |
  |          | Atoms |           |
  |          ++-----++           |
  |           |     |            |
  |        +--+    ++------+     |
  |        |       |Derived|     |
  |        v       ++-----++     |
  |    +----+       |     |      |
  |    |Term|       v     v      |
  |    +---++   +----+ +----+    |
  |        |    |Term| |Term|    |
  |        v    +--+-+ +--+-+    |
  |      +-----+   |      |      |
  |      |Total|<--+------+      |
  |      +-----+                 |
  |                              |
  +------------------------------+

