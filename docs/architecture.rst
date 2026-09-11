.. _architecture:

Architecture
============

tmol expresses molecular modeling as batched tensor operations integrated with
PyTorch. The principal representation is ``tmol.pose.PoseStack``. It replaces
the historical ``tmol.system`` score-graph interface described in older versions
of this page. Independent structures can have different sizes, sequences and
chemistries; they do not interact simply because they share a batch.

Representation and setup
------------------------

``PoseStack.coords`` has shape ``[n_poses, max_n_atoms, 3]``. Block-type indices,
atom offsets and inter-block connections encode each structure's layout.
A block can represent a polymer residue, a non-polymer ligand, or a connected
fragment of a larger component. ``PackedBlockTypes`` stores the active block
types in a shared table indexed by each block occurrence. Chemistry-specific
annotations are computed once where reusable, then packed into device tensors.
See `PoseStack`_ and `Packed block types`_.

``ParameterDatabase`` separates chemical definitions from score-term parameter
tables. Preparation returns a new database instead of mutating the input.
``PoseBuildContext`` groups compatible construction objects for reuse.
The frozen chemical records coexist with derived caches on refined types and
packed tables; immutable database does not mean every implementation object
is immutable. Preserve database identity and configuration when reusing caches.

The repeated numerical path
---------------------------

A ``ScoreFunction`` owns weights and energy-term objects. Rendering performs
block-type, packed-type and pose-topology setup before producing whole-pose,
block-pair or rotamer scoring modules. Whole-pose scoring returns one weighted
energy per pose by default; term-resolved and positional outputs support
diagnostics. Compatible terms reuse low-level functions across these modes.
Rosetta also shares energy methods between protocols; tmol's difference is the
tensorized setup and execution contract. See `Score function`_.

Compiled operators expose coordinate derivatives through PyTorch autograd.
For a learned coordinate generator :math:`x=f_\theta(z)`, differentiable energy
evaluation provides

.. math::

   \frac{\partial E}{\partial\theta}
   = \left(\frac{\partial f_\theta}{\partial\theta}\right)^T
     \frac{\partial E}{\partial x}.

The coordinate generator must preserve its autograd graph. Chemical typing,
discrete rotamer selection and the entire FastRelax control flow are not
thereby differentiable, and higher derivatives require separate validation.
Coordinate-only changes can reuse a rendered module. Changes to layout,
topology, active types or score configuration require appropriate setup and
rerendering.

The LJ/LK CUDA whole-pose kernel uses block-pair work units, bounding-sphere
culling and 32-atom shared-memory tiles. A 32-thread warp processes a surviving
block pair and reduces its interaction contributions. CPU and CUDA paths share
low-level functional expressions with device-specific dispatch; other score
terms can use different algorithms. This arrangement enables batching and data
reuse but does not by itself establish a speedup on a particular workload.
See `LJLK kernel`_.

Kinematics and combinatorial optimization
-----------------------------------------

``FoldForest`` describes polymer, jump, root-jump and chemical edges and is
expanded into an atom-level ``KinForest``. Dependency-ordered segmented scans
compose local transforms and propagate derivative information. For transforms
:math:`T_i`, the frame along a path is
:math:`X_j=X_0\prod_{i=1}^{j}T_i`; associativity permits a prefix scan within
each independent segment. Branches are scheduled after their dependencies.

The convenience ``reasonable_fold_forest`` constructor currently uses host
NumPy arrays to discover edges, and conjugated-group discovery also walks
host-side topology. The numerical kinematic operators and portions of schedule
construction are tensorized/compiled. Do not describe the entire chemistry-to-
coordinates pipeline as GPU-resident. Cyclic bonds excluded from the kinematic
spanning tree remain distinct chemical constraints. See `Fold forest`_ and
`Kinematic kernels`_.

Packing builds candidate coordinates and compatible one-/two-block energies
into a sparse interaction graph. CPU and CUDA annealers search the discrete
choices. Covalent-group packing adds a common conformer index and reduces the
group to one representative choice without merging its chemical block records.
FastRelax alternates packing with minimization under a repulsive-weight
schedule. See :ref:`chemistry-workflows` for the necessary sampling configuration.

Related guides
--------------

* :ref:`noncanonical-chemistry`: chemical input, preparation and supported scope.
* :ref:`chemistry-workflows`: API recipes and sampler choices.
* :ref:`rosetta-comparison`: inherited methods, concrete differences and limits.

.. include:: _source_links.inc
