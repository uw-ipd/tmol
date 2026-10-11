Tutorials
=========

Runnable notebooks with molecular viewers, plots, and exercises. Start with
01–03, then use packing (04) and minimization (05) before FastRelax (06).
For a specific operation, use the :doc:`task index <tutorial/recipe_index>`.

In Colab, select **Runtime → Change runtime type → T4 GPU**, then **Run all**.
The setup cell installs TMol and downloads the example data. The bootstrap
supports Python 3.12/3.13, PyTorch 2.11, and CUDA 12.8, including Colab's
**2026.07** runtime. For local execution, follow :doc:`installation`.

.. nbgallery::
   :caption: Basics

   tutorial/01_working_with_tmol
   tutorial/02_gpu_batching
   tutorial/03_scoring_and_analysis
   tutorial/04_packing_and_mutation_scan
   tutorial/05_minimization_constraints_kinematics
   tutorial/06_fast_relax
   tutorial/07_ligand_and_params
   tutorial/08_nucleic_acids

.. nbgallery::
   :caption: Applications

   tutorial/09_protein_interface_hotspot_scan
   tutorial/10_ligand_pose_sensitivity

.. nbgallery::
   :caption: Extending TMol

   tutorial/11_extending_chemistry_and_scoring
   tutorial/12_explicit_foldforests_and_torsions
   tutorial/13_extending_the_packer

.. toctree::
   :hidden:

   Task index <tutorial/recipe_index>
