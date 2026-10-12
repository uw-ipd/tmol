Tutorials
=========

Runnable notebooks with molecular viewers, plots, and exercises. Start with
01–03, then use packing (04) and minimization (05) before FastRelax (06).
For a specific operation, use the :doc:`task index <tutorial/recipe_index>`.

In Colab, select a GPU under **Runtime → Change runtime type**, then **Run all**.
The shared setup installs a tested TMol release with operators matching Colab's
PyTorch CUDA major and downloads the example data. For local notebooks and
supported environments, follow :doc:`installation`.

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
