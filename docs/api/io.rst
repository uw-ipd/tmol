Input and Output
================

The public :mod:`tmol.io` API converts common structure representations to and
from :class:`tmol.pose.PoseStack`. The direct AtomWorks adapter uses its
protein-only unified Atom37 representation. For differentiable Atom37
coordinates with general Biotite topology, including nucleic acids and ligands,
use :func:`tmol.io.pose_stack_from_atom37_and_biotite`. Repeated diffusion,
guidance, and search workloads should bind their fixed topology once with
:func:`tmol.io.prepare_pose_stack_from_atom37`; its returned callable accepts
each coordinate batch and optimizes hydrogens by default. For backbone-only
input -- N/CA/C/O with no side chains, as produced by backbone generators --
use :func:`tmol.io.pose_stack_from_backbone_coords`, which completes the
missing side chains with the packer.

.. automodule:: tmol.io
   :members:
   :imported-members:
   :show-inheritance:
