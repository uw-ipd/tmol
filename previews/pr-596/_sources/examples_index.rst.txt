Tutorials
=========

Each notebook works through a molecular experiment, with code, plots, and
structure viewers together. Run it independently from top to bottom.
Start with loading (01) and scoring (03); use batching (02) when you want to
process several structures. Then choose the operation or molecular system
you need below. The :doc:`task index <tutorial/recipe_index>` lists individual
API recipes.

Setup and lengthy plotting helpers start folded. Expand them when you need
the details; the modeling steps and their results stay visible. Sections
marked **Optional** extend a completed example.

In Colab, select a GPU under **Runtime → Change runtime type**, then **Run all**.
The shared setup installs a tested TMol release with operators matching Colab's
PyTorch CUDA major and downloads the example data. For local notebooks and
supported environments, follow :doc:`installation`.

.. nbgallery::
   :caption: Load and score structures

   tutorial/01_working_with_tmol
   tutorial/02_gpu_batching
   tutorial/03_scoring_and_analysis

.. nbgallery::
   :caption: Change conformations and sequence

   tutorial/04_packing_and_mutation_scan
   tutorial/05_minimization_constraints_kinematics
   tutorial/06_fast_relax

.. nbgallery::
   :caption: Ligands, nucleic acids, and interfaces

   tutorial/07_ligand_and_params
   tutorial/08_nucleic_acids
   tutorial/09_protein_interface_hotspot_scan
   tutorial/10_ligand_pose_sensitivity

.. nbgallery::
   :caption: Extend TMol

   tutorial/11_extending_chemistry_and_scoring
   tutorial/12_explicit_foldforests_and_torsions
   tutorial/13_extending_the_packer

.. toctree::
   :hidden:

   Task index <tutorial/recipe_index>
