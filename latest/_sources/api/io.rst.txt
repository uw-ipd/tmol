Input and Output
================

The public :mod:`tmol.io` API converts common structure representations to and
from :class:`tmol.pose.PoseStack`. The direct AtomWorks adapter uses its
protein-only unified Atom37 representation. For differentiable Atom37
coordinates with general Biotite topology, including nucleic acids and ligands,
use :func:`tmol.io.pose_stack_from_atom37_and_biotite`.

.. automodule:: tmol.io
   :members:
   :imported-members:
   :show-inheritance:
