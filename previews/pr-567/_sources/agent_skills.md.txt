# Agent skills

The `skills/` directory contains instructions for coding agents. Each
`SKILL.md` covers one task and links to the corresponding docs and APIs.

| Skill | Use it for |
| --- | --- |
| `tmol-setup` | Select a wheel or source/JIT install, verify CPU/CUDA support, and diagnose extension loading. |
| `tmol-score` | Load a structure, render a scorer, inspect terms, compute coordinate gradients, and choose CPU or CUDA execution. |
| `tmol-pack-relax` | Configure repacking or design, minimize coordinates, and run FastRelax with bounded CPU/GPU memory. |
| `tmol-develop` | Build extensions, run focused CPU/CUDA tests, profile kernels, and prepare a reviewable performance change. |

## Install or vendor a skill

Install a skill from GitHub or copy `skills/<name>/` into your agent's skill
search path. See
[`skills/README.md`](https://github.com/uw-ipd/tmol/tree/master/skills) for the
catalog and artifact conventions.

Skills do not authorize external jobs, compute spending, pushes, or merges;
those require user authorization.
