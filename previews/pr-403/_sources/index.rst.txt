tmol documentation
==================

``tmol`` is a GPU-accelerated PyTorch implementation of Rosetta's molecular
modeling energy function. It scores protein structures, propagates derivatives
through coordinates, packs side chains, minimizes structures, and prepares
small-molecule ligands for protein-ligand scoring workflows.

Start with :doc:`quickstart` for a first score calculation. The guide pages
then cover installation, guided tutorials, workflows, runnable examples, API
details, and contributor workflows.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   quickstart
   installation
   tutorial/index
   workflows/index
   examples_index
   architecture
   datatypes

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api_reference
   contributor_guide
