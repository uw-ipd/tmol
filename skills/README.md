# TMol Agent Skills

Portable agent skills for common TMol user and contributor workflows. Each
skill is a self-contained directory with an agent-facing `SKILL.md`, a
human-readable `skill-card.md`, and a bounded case under `evals/`. They encode
TMol's real public APIs, build commands, and execution constraints so an agent
does not have to rediscover them from implementation details.

## Catalog

| Skill | What it does | Typical order |
| --- | --- | --- |
| [`tmol-setup`](tmol-setup/) | Choose a compatible wheel or source/JIT installation and verify CPU/CUDA extension loading. | 1 |
| [`tmol-score`](tmol-score/) | Load and score structures, inspect terms, compute gradients, and configure CPU threads or CUDA graph replay. | 2 |
| [`tmol-pack-relax`](tmol-pack-relax/) | Repack or design side chains, minimize coordinates, and run FastRelax with bounded memory. | 3 |
| [`tmol-develop`](tmol-develop/) | Build, test, benchmark, profile, and review TMol implementation changes. | Contributor workflow |

Typical modeling order is **setup → score → pack/relax**. The development skill
is independent and applies to repository changes rather than ordinary library
use.

## Use

Each skill can be vendored by copying its directory into an agent environment's
skill search path. Environments that discover skills directly from a repository
can point at `skills/<name>/SKILL.md`.

The skill text and helper artifacts use TMol's top-level Apache-2.0 license.
Generated signatures or benchmark reports, if required by a downstream skill
catalog, are onboarding artifacts and are not maintained in this repository.
