Tutorials
=========

These eight executable notebooks are in-depth demonstrations with live molecular
viewers, selectable structures and atom subsets, result tables, plots, and
exercises. For shorter, reusable recipes, see :doc:`Workflows <workflows/index>`.
Start with Tutorials 01–03. From scoring, branch to packing (04) or
minimization (05), then combine both in FastRelax (06). Tutorials 07 and 08
cover specialized ligand and nucleic-acid workflows. The
:doc:`learning-path guide <learning_paths>` explains the sequence.

Use the :doc:`task index <tutorial/recipe_index>` to find a maintained
tutorial, workflow recipe, or API page for a specific operation.

Readers coming from Rosetta or PyRosetta can consult the
:doc:`Rosetta-to-TMol crosswalk <tutorial/rosetta_crosswalk>` alongside the
Tutorials. The crosswalk distinguishes genuine API parallels from capabilities
that tmol does not currently implement.

.. raw:: html

   <div class="example-card-grid">
     <a class="example-card" href="tutorial/01_working_with_tmol.html">
       <img src="_static/tutorials/01_working_with_tmol.svg" alt="1UBQ with residues 1 through 10 highlighted">
       <span class="example-card-title">01. Working with TMol</span>
       <span class="example-card-description">CIF and PDB input, preparation choices, PoseStack concepts, export, and visualization.</span>
     </a>
     <a class="example-card" href="tutorial/02_gpu_batching.html">
       <img src="_static/tutorials/02_gpu_batching.svg" alt="1UBQ, 1R21, and 1BL8 arranged as one heterogeneous structure batch">
       <span class="example-card-title">02. GPU batching</span>
       <span class="example-card-description">Reliable CUDA timing, throughput, memory, chunking, and PyRosetta parallelism.</span>
     </a>
     <a class="example-card" href="tutorial/03_scoring_and_analysis.html">
       <img src="_static/tutorials/03_scoring_and_analysis.svg" alt="1UBQ residues 1 through 30 with the analyzed residue pair highlighted">
       <span class="example-card-title">03. Scoring and analysis</span>
       <span class="example-card-description">Score terms, block-pair matrices, differentiable coordinates, and output analysis.</span>
     </a>
     <a class="example-card" href="tutorial/04_packing_and_mutation_scan.html">
       <img src="_static/tutorials/04_packing_and_mutation_scan.svg" alt="Ten-residue 1UBQ slice with the local packing shell and mutation target marked">
       <span class="example-card-title">04. Packing and mutation scans</span>
       <span class="example-card-description">Regional repacking, conformer samplers, and a carefully scoped mutation-score proxy.</span>
     </a>
     <a class="example-card" href="tutorial/05_minimization_constraints_kinematics.html">
       <img src="_static/tutorials/05_minimization_constraints_kinematics.svg" alt="Restrained eight-residue 1UBQ minimization system">
       <span class="example-card-title">05. Minimization and kinematics</span>
       <span class="example-card-description">Cartesian and torsional minimization, low-level constraints, MoveMaps, and explicit FoldForests.</span>
     </a>
     <a class="example-card" href="tutorial/06_fast_relax.html">
       <img src="_static/tutorials/06_fast_relax.svg" alt="Six-residue 1UBQ system beside its two-stage pack-minimize schedule">
       <span class="example-card-title">06. FastRelax</span>
       <span class="example-card-description">Repack/minimize schedules, score-weight ramps, and structure comparison.</span>
     </a>
     <a class="example-card" href="tutorial/07_ligand_and_params.html">
       <img src="_static/tutorials/07_ligand_and_params.svg" alt="ADA ligand and complete protein residues in its 4.5 angstrom pocket">
       <span class="example-card-title">07. Ligands and parameter files</span>
       <span class="example-card-description">CIF ligand chemistry, Rosetta .params, TMol YAML, selections, and interaction scores.</span>
     </a>
     <a class="example-card" href="tutorial/08_nucleic_acids.html">
       <img src="_static/tutorials/08_nucleic_acids.svg" alt="1HDD protein-DNA complex and enlarged 1EHT RNA-theophylline aptamer">
       <span class="example-card-title">08. Working with DNA and RNA</span>
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
