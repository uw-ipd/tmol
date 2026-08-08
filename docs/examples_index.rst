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
       <img src="_static/tutorials/01_working_with_tmol.svg" alt="CIF input converted into a molecular PoseStack">
       <span class="example-card-title">1. Working with TMol</span>
       <span class="example-card-description">CIF input, PoseStack concepts, AtomWorks selections, and interactive visualization.</span>
     </a>
     <a class="example-card" href="tutorial/02_gpu_batching.html">
       <img src="_static/tutorials/02_gpu_batching.svg" alt="Molecular batches flowing through a GPU">
       <span class="example-card-title">2. GPU batching</span>
       <span class="example-card-description">Reliable CUDA timing, throughput, memory, chunking, and PyRosetta parallelism.</span>
     </a>
     <a class="example-card" href="tutorial/03_scoring_and_analysis.html">
       <img src="_static/tutorials/03_scoring_and_analysis.svg" alt="Block-pair score heatmap and molecular contact">
       <span class="example-card-title">3. Scoring and analysis</span>
       <span class="example-card-description">Score terms, block-pair matrices, differentiable coordinates, and output analysis.</span>
     </a>
     <a class="example-card" href="tutorial/04_packing_and_mutation_scan.html">
       <img src="_static/tutorials/04_packing_and_mutation_scan.svg" alt="Side-chain rotamers and amino-acid substitutions">
       <span class="example-card-title">4. Packing and mutation scans</span>
       <span class="example-card-description">Regional repacking, conformer samplers, and a carefully scoped mutation-score proxy.</span>
     </a>
     <a class="example-card" href="tutorial/05_minimization_constraints_kinematics.html">
       <img src="_static/tutorials/05_minimization_constraints_kinematics.svg" alt="FoldForest edges and restrained atoms">
       <span class="example-card-title">5. Minimization and kinematics</span>
       <span class="example-card-description">Cartesian and torsional minimization, constraints, MoveMaps, and FoldForests.</span>
     </a>
     <a class="example-card" href="tutorial/06_fast_relax.html">
       <img src="_static/tutorials/06_fast_relax.svg" alt="FastRelax pack-minimize trajectory">
       <span class="example-card-title">6. FastRelax</span>
       <span class="example-card-description">Repack/minimize schedules, score-weight ramps, and structure comparison.</span>
     </a>
     <a class="example-card" href="tutorial/07_ligand_and_params.html">
       <img src="_static/tutorials/07_ligand_and_params.svg" alt="Ligand pocket with TMol and Rosetta parameter documents">
       <span class="example-card-title">7. Ligands and parameter files</span>
       <span class="example-card-description">CIF ligand chemistry, Rosetta .params, TMol YAML, selections, and interaction scores.</span>
     </a>
     <a class="example-card" href="tutorial/08_nucleic_acids.html">
       <img src="_static/tutorials/08_nucleic_acids.svg" alt="DNA helix, protein binder, and RNA ligand pocket">
       <span class="example-card-title">8. Working with DNA and RNA</span>
       <span class="example-card-description">NA scoring and rotamers, homeodomain–DNA base substitutions, and a fixed-ligand RNA aptamer.</span>
     </a>
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
   tutorial/08_nucleic_acids
   tutorial/rosetta_crosswalk
