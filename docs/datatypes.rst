.. _datatypes:

==========================
Data model and conventions
==========================

TMol combines PyTorch tensors with immutable or functionally updated Python
data objects. These conventions make batched molecular state explicit and keep
device placement visible to callers.

Tensor shapes
=============

Numeric data is generally stored in :class:`torch.Tensor` objects. Leading
dimensions describe molecular axes such as poses, blocks, atoms, score terms,
or rotamers. Common examples include:

.. list-table::
   :header-rows: 1

   * - Data
     - Typical shape
     - Meaning
   * - Pose coordinates
     - ``[n_poses, max_n_atoms, 3]``
     - Cartesian coordinates with padding for heterogeneous systems.
   * - Block types
     - ``[n_poses, max_n_blocks]``
     - Index of each block's chemical type, with sentinel padding.
   * - Whole-pose scores
     - ``[n_poses]`` or ``[n_terms, n_poses]``
     - Weighted totals or a score-term decomposition.
   * - Block-pair scores
     - ``[n_poses, n_blocks, n_blocks]``
     - Directed block-pair accounting for analysis and reweighting.

Boolean masks identify real atoms or selected coordinates. Integer tensors
encode topology, block types, connections, and kinematic indices. Do not infer
valid entries from coordinate values alone.

Device and dtype
================

All tensors participating in one operation must use compatible devices and
dtypes. A :class:`~tmol.pose.PoseStack`, its
:class:`~tmol.pose.PackedBlockTypes`, rendered scoring modules, and movement
data normally live on the same CPU or CUDA device.

TMol APIs commonly accept a device-like value and normalize it through
:func:`tmol.utility.resolve_device`. Preserve the input coordinate dtype unless
an API documents a stronger requirement; silently mixing ``float32`` and
``float64`` changes performance and can invalidate comparisons.

Python data objects
===================

TMol uses ``attrs``-based classes and typed containers for structured metadata,
database records, I/O contexts, and protocol configuration. Many of these
objects are treated as immutable: extension operations return a new object
instead of mutating process-global state.

NumPy arrays are primarily used where third-party libraries require them or
where multidimensional string/object data does not fit a Torch tensor. Convert
deliberately at library boundaries and remember that moving through NumPy
breaks PyTorch autograd.

Public typing helpers
=====================

The :mod:`tmol.types` package provides runtime conversion, validation, tensor
shape annotations, and ``TensorGroup`` helpers used throughout the codebase.
The :mod:`tmol.utility` package contains device, cumulative-sum, units, and
other implementation helpers. Most workflow users interact with the molecular
objects in :mod:`tmol.pose`, :mod:`tmol.io`, and :mod:`tmol.score` instead.

See :doc:`terminology` for the difference between poses, blocks, deposited
atoms, and built atoms.
