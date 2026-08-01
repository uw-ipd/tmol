.. _architecture:

============
Architecture
============

TMol is centered on two core representations: batched molecular state in
``tmol.pose`` and term-specific scoring machinery in ``tmol.score``. Structure
I/O in ``tmol.io`` builds :class:`~tmol.pose.pose_stack.PoseStack` objects from
PDB/mmCIF files and model outputs.

These components operate over a shared chemical vocabulary defined in
``tmol.database.chemical``, with additional term-specific data given in
``tmol.database.scoring``.

.. code-block:: text

  +------+       +------+          +---------+
  |      |       |      |          |         |
  |  io  +------>+ pose +----------o scoring |
  |      |       |      |          |         |
  +------+       +--+---+          +--+----+-+
                    |                 |    |
                    | +---------------v-+  |
                    | |                 |  |
                    | | database.scoring|  |
                    | |                 |  |
                    | +--------+--------+  |
                    |          |           |
                    | +--------v--------+  |
                    | |                 |  |
                    +->database.chemical<-+
                      |                 |
                      +-----------------+

``tmol.pose`` and ``tmol.score`` meet when a
:class:`~tmol.score.score_function.ScoreFunction` renders a scoring module for a
:class:`~tmol.pose.pose_stack.PoseStack`, for example with
:meth:`~tmol.score.score_function.ScoreFunction.render_whole_pose_scoring_module`.
Score terms annotate :class:`~tmol.pose.packed_block_types.PackedBlockTypes`
and then render ``torch.nn.Module`` objects for repeated evaluation.

Scoring Overview
================

Scoring is managed by rendered PyTorch modules that evaluate configured energy
terms over one or more poses. A model is defined over a set ``n`` of bonded
atoms. Each atom is located at an atom index and is defined by a type and
coordinate. Atoms may be null, defining no type and a NaN coordinate at a given
index. Bonds are represented as sparse, undirected bonded inter-atom index
pairs.

.. code-block:: text

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
Intra-layer scoring is defined across all interactions, bonded and non-bonded,
within a layer. Inter-layer scoring is defined over inter-layer non-bonded
interactions.

.. note:: ``tmol.score`` currently only supports intra-layer scoring and is
   limited to models of depth 1.

The score function implementation is partitioned into score term classes, each
covering a logically distinct component of the energy function. These
components include score terms, derived pose representations, and support data
required for score evaluation. A rendered scoring module includes an atomic
representation, some number of score terms, and a weighted total score.

.. code-block:: text

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
