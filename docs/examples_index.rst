Interactive tutorials
=====================

These CIF-first notebooks form one guided path through tmol. They use live
molecular viewers, selectable structures and atom subsets, sortable result
tables, plots, and executable exercises. Start with the first two notebooks;
later notebooks assume their ``PoseStack`` and GPU-batching vocabulary.

Readers coming from Rosetta or PyRosetta can consult the
:doc:`Rosetta-to-TMol crosswalk <tutorial/rosetta_crosswalk>` alongside the
tutorials. The crosswalk distinguishes genuine API parallels from capabilities
that tmol does not currently implement.

.. raw:: html

   <div class="example-card-grid">
     <a class="example-card" href="tutorial/01_working_with_tmol.html">
       <img src="_static/tutorial_thumbnail.svg" alt="TMol molecular structure tutorial">
       <span class="example-card-title">1. Working with TMol</span>
       <span class="example-card-description">CIF input, PoseStack concepts, AtomWorks selections, and interactive visualization.</span>
     </a>
     <a class="example-card" href="tutorial/02_gpu_batching.html">
       <img src="_static/tutorial_thumbnail.svg" alt="TMol GPU batching tutorial">
       <span class="example-card-title">2. GPU batching</span>
       <span class="example-card-description">Reliable CUDA timing, throughput, memory, chunking, and PyRosetta parallelism.</span>
     </a>
     <a class="example-card" href="tutorial/03_scoring_and_analysis.html">
       <img src="_static/tutorial_thumbnail.svg" alt="TMol scoring tutorial">
       <span class="example-card-title">3. Scoring and analysis</span>
       <span class="example-card-description">Score terms, block-pair matrices, differentiable coordinates, and output analysis.</span>
     </a>
     <a class="example-card" href="tutorial/04_packing_and_mutation_scan.html">
       <img src="_static/tutorial_thumbnail.svg" alt="TMol packing tutorial">
       <span class="example-card-title">4. Packing and mutation scans</span>
       <span class="example-card-description">Regional repacking, conformer samplers, and a carefully scoped mutation-score proxy.</span>
     </a>
     <a class="example-card" href="tutorial/05_minimization_constraints_kinematics.html">
       <img src="_static/tutorial_thumbnail.svg" alt="TMol minimization tutorial">
       <span class="example-card-title">5. Minimization and kinematics</span>
       <span class="example-card-description">Cartesian and torsional minimization, constraints, MoveMaps, and FoldForests.</span>
     </a>
     <a class="example-card" href="tutorial/06_fast_relax.html">
       <img src="_static/tutorial_thumbnail.svg" alt="TMol FastRelax tutorial">
       <span class="example-card-title">6. FastRelax</span>
       <span class="example-card-description">Repack/minimize schedules, score-weight ramps, and structure comparison.</span>
     </a>
     <a class="example-card" href="tutorial/07_ligand_and_params.html">
       <img src="_static/tutorial_thumbnail.svg" alt="TMol ligand parameter tutorial">
       <span class="example-card-title">7. Ligands and parameter files</span>
       <span class="example-card-description">CIF ligand chemistry, Rosetta .params, TMol YAML, selections, and interaction scores.</span>
     </a>
     <div class="example-card example-card-pending">
       <img src="_static/tutorial_thumbnail.svg" alt="TMol nucleic-acid tutorial coming after PR 404">
       <span class="example-card-title">8. DNA and RNA — after PR #404</span>
       <span class="example-card-description">Added after the nucleic-acid APIs are present so every published cell remains executable.</span>
     </div>
   </div>

.. toctree::
   :maxdepth: 1
   :hidden:

   tutorial/01_working_with_tmol
   tutorial/02_gpu_batching
   tutorial/03_scoring_and_analysis
   tutorial/04_packing_and_mutation_scan
   tutorial/05_minimization_constraints_kinematics
   tutorial/06_fast_relax
   tutorial/07_ligand_and_params
   tutorial/rosetta_crosswalk

Additional examples
-------------------

The shorter examples below remain useful as compact API references.

.. toctree::
   :maxdepth: 1

   tutorial/visualisation
   tutorial/score_pack_minimize
