.. _architecture:

============
Architecture
============

TMol is centered on two core representations: batched molecular state in
``tmol.pose`` and term-specific scoring machinery in ``tmol.score``. Structure
I/O in ``tmol.io`` builds :class:`~tmol.pose.PoseStack` objects from
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
:class:`~tmol.score.ScoreFunction` renders a scoring module for a
:class:`~tmol.pose.PoseStack`, for example with
:meth:`~tmol.score.ScoreFunction.render_whole_pose_scoring_module`.
Score terms annotate :class:`~tmol.pose.PackedBlockTypes`
and then render ``torch.nn.Module`` objects for repeated evaluation.

Scoring Overview
================

Scoring is managed by rendered PyTorch modules that evaluate configured energy
terms over a ``PoseStack``. Coordinates have shape
``[n_poses, max_n_atoms, 3]``; ``real_atoms`` distinguishes molecular atoms
from padding, while block-type and connection tensors describe residue and
polymer topology.

.. code-block:: text

  PoseStack + ScoreFunction
             |
             +--> whole-pose module --> [n_poses]
             |
             +--> block-pair module --> [n_poses, n_blocks, n_blocks]
             |
             +--> rotamer module -----> packer candidate energies

The score function implementation is partitioned into score term classes, each
covering a logically distinct component of the energy function. These
term annotates residue/block data before rendering its coordinate-dependent
module. Calls may return either the weighted total or a leading score-term axis
when ``sum_terms=False``. The complete score-type-to-term map is documented in
:doc:`api/score_terms`.
