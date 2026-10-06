.. _architecture:

============
Architecture
============

``tmol.io`` builds batched :class:`~tmol.pose.PoseStack` objects from structures
or model tensors. ``tmol.score`` evaluates their coordinates using chemical
definitions from ``tmol.database.chemical`` and parameters from
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

Modeling lifecycle
==================

Structure preparation, scoring, and output follow this sequence:

.. code-block:: text

  structure records or model tensors
                 |
                 v
        I/O and chemical typing
                 |
                 v
      PoseStack + build context
          |              |
          v              v
   rendered scorers   packing / movement setup
          |              |
          +-------+------+
                  v
       score, optimize, or analyze
                  |
                  v
       Biotite structure or PDB output

The :class:`~tmol.database.ParameterDatabase` supplies chemical and scoring
definitions. I/O chooses compatible block types and constructs a
:class:`~tmol.pose.PackedBlockTypes` collection on the requested device. The
resulting :class:`~tmol.pose.PoseStack` owns coordinates, topology, block
indices, and references to those packed types.

Packing may return a new stack when chemical identities or atom counts change.
Cartesian or kinematic minimization usually changes coordinates while keeping
the same layout. That distinction determines whether an existing rendered
scoring module can be reused.

``tmol.pose`` and ``tmol.score`` meet when a
:class:`~tmol.score.ScoreFunction` renders a scoring module for a
:class:`~tmol.pose.PoseStack`, for example with
:meth:`~tmol.score.ScoreFunction.render_whole_pose_scoring_module`.
Score terms annotate :class:`~tmol.pose.PackedBlockTypes`
and then render ``torch.nn.Module`` objects for repeated evaluation.

Scoring overview
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
covering a logically distinct component of the energy function. Each term
annotates residue and block data before rendering its coordinate-dependent
module. Calls may return either the weighted total or a leading score-term axis
when ``sum_terms=False``. The complete score-type-to-term map is documented in
:doc:`api/score_terms`.

See :doc:`datatypes` for tensor conventions and :doc:`api/score_terms` for
individual score terms.
