# Executable examples

- [Example 01: score a PDB file](example_01_score_a_pdb_file.ipynb)
- [Example 02: build your own model input interface](example_02_model_inputs.ipynb) — OpenFold-style atom14, RF2 hydrogen policies, scoring and coordinate gradients, and repeated guidance.

[![Open Example 02 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kierandidi/tmol/blob/review/pr503-chemistry-efficiency/notebooks/example_02_model_inputs.ipynb)

Example 02 installs pinned development sources against the runtime's PyTorch. The unpublished AtomWorks dependency currently needs a Colab secret named `ATOMWORKS_TOKEN` with read access to `baker-laboratory/atomworks-dev`. Replace this source dependency with a public release once the companion PR lands. GPU execution needs a CUDA toolkit compatible with the runtime's PyTorch; CPU execution is also supported. Native extensions compile on first use.

In an existing tmol development environment, set `TMOL_EXAMPLE_SOURCE` to the checkout and optionally `TMOL_EXAMPLE_DEVICE=cpu` or `cuda`. This reuses that environment; every subsequent notebook cell is identical. The pytest workflow executes those cells on CPU and CUDA and checks exact supplied C-alpha coordinates, finite scores/gradients, RF2 rebuilt-slot gradients, and repeated fixed-topology guidance.
