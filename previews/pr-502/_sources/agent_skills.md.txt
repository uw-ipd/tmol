# Agent skills

TMol includes portable agent skills for recurring user and contributor
workflows. Each skill is a directory under `skills/` with a `SKILL.md` that an
agent reads only when the request matches that workflow. The skills point to
the maintained documentation and public APIs rather than duplicating them.

| Skill | Use it for |
| --- | --- |
| `tmol-setup` | Select a wheel or source/JIT install, verify CPU/CUDA support, and diagnose extension loading. |
| `tmol-score` | Load a structure, render a scorer, inspect terms, compute coordinate gradients, and choose CPU or CUDA execution. |
| `tmol-pack-relax` | Configure repacking or design, minimize coordinates, and run FastRelax with bounded CPU/GPU memory. |
| `tmol-develop` | Build extensions, run focused CPU/CUDA tests, profile kernels, and prepare a reviewable performance change. |

## Install or vendor a skill

The repository directories are self-contained. An agent environment that
supports repository skills can install one from the GitHub repository or vendor
the corresponding `skills/<name>/` directory into its skill search path. Read
[`skills/README.md`](https://github.com/uw-ipd/tmol/tree/master/skills) for the
catalog and artifact conventions.

Skills do not grant permission to submit jobs, spend external compute, push
branches, or merge pull requests. Those actions still require the authority
provided by the user's request and the active environment.
