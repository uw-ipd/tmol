tmol documentation
==================

``tmol`` is a PyTorch library with Rosetta-inspired all-atom score terms. It
scores structures on CPU or GPU, supports autograd, side-chain packing,
minimization, and ligand preparation. Weighted outputs are TMol score units,
not kcal/mol or calibrated Rosetta score units, and are not guaranteed to match
Rosetta numerically.

Start with :doc:`quickstart` for a first score calculation. The guide pages
then cover installation, workflows, examples, API details, and contribution
workflows.

.. toctree::
   :maxdepth: 1

   quickstart
   installation
   workflows/index
   examples_index
   api_reference
   Contributing <contributor_guide>
