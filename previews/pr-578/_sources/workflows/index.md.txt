# Guides

Examples and API details for common modeling tasks. Start with the
{doc}`Quickstart </quickstart>` if you have not used TMol before.

```{toctree}
:maxdepth: 1
:caption: Structures and tensors

../concepts
structure_io
../user_guide/integrations
../model_inputs
gpu_batching
../user_guide/cpu_threading
```

```{toctree}
:maxdepth: 1
:caption: Scoring and modeling

../user_guide/scoring
packing
../user_guide/optimization
protein_interfaces
../user_guide/ligands
nucleic_acids
```

```{toctree}
:maxdepth: 1
:caption: Extension and performance

extending_tmol
../user_guide/benchmarking
```

Packing can change block types and atom counts: render a new scorer for its
output. Coordinate-only minimization can reuse a scorer for the same layout.
See {doc}`Scorer reuse </terminology>` for details.
