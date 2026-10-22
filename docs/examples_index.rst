Tutorials
=========

The tutorials are a how-to guide to TMol, split into runnable notebooks. Each
section works through its operations with code, results, and structure viewers
side by side, and can be run independently from top to bottom. Start with
loading structures (1), then choose the operation or molecular system you need
below. Sections marked **TODO** are still being written.

Setup starts folded. Expand it when you need the details; the modeling steps and
their results stay visible.

In Colab, select a GPU under **Runtime → Change runtime type**, then **Run all**.
The shared ``notebook_setup`` module installs a tested TMol release with
operators matching Colab's PyTorch CUDA major and downloads the example data.
For local notebooks and supported environments, follow :doc:`installation`.

Prefer one continuous walkthrough over switching between notebooks? The
sections below are also available combined into a single notebook, sharing
one Setup cell at the top.

.. nbgallery::
   :caption: Complete walkthrough

   tutorial/tmol_tutorial

.. nbgallery::
   :caption: Load, score, and optimize

   tutorial/01_tmol_how_to_guide
   tutorial/02_scoring_and_optimization

.. nbgallery::
   :caption: Metals, nucleic acids, ligands, and NCAAs

   tutorial/03_metals_and_nucleic_acids
   tutorial/04_ligands_and_ncaas

.. nbgallery::
   :caption: Selections, constraints, and advanced topics

   tutorial/05_selections
   tutorial/06_constraints
   tutorial/07_advanced_kinematics
   tutorial/08_advanced_topics
