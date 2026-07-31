tmol documentation
==================

``tmol`` is a GPU-accelerated PyTorch implementation of Rosetta's molecular
modeling energy function. It scores protein structures, propagates derivatives
through coordinates, packs side chains, minimizes structures, and prepares
small-molecule ligands for protein-ligand scoring workflows.

Start with :doc:`quickstart` for a first score calculation. The user guide
then covers installation, ligand preparation, optimization, integrations, and
development.

.. toctree::
   :maxdepth: 2
   :caption: Get Started

   quickstart
   installation

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/scoring
   user_guide/ligands
   user_guide/optimization
   user_guide/integrations
   user_guide/benchmarking
   user_guide/development

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index
   notebooks/tmol_how_to_guide

.. toctree::
   :maxdepth: 2
   :caption: Reference

   architecture
   datatypes
   api_reference
   contributor_guide
