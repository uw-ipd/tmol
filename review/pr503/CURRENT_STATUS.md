# Review checkpoint: 2026-09-12

Frank's latest fetched #503 head is `0593a93b07d80b0302383163d2d98c78e315ab98`,
already an ancestor of this branch. Production at `01159dd7a` completed the
broad and corpus runs. The final extra reader rejection test passes locally
and with published AtomWorks; it does not change production behavior.

- Broad selection: 1,459 passed, 15 skipped, eight existing noncanonical score
  reference failures. All 18 CPU and 18 CUDA examples pass.
- Original local corpus: tmol 14 pass / 2 partial / 17 fail; AtomWorks 9 / 2 / 22.
- Explicit free-metal-excluded diagnostic: tmol 16 / 6 / 11; AtomWorks 10 / 7 / 16.
- Released AtomWorks 2.2.1 matrix: 18 of 19 pass scoring/gradient/rotamer checks;
  observed 8OG OP2 deletion is rejected. Five reader tests pass in both versions.
- Numerical passes do not imply convergence or physical validation. Four
  original-input trials converged; 6mub retains glycan-link geometry alerts.

See [CORPUS_FINDINGS.md](CORPUS_FINDINGS.md),
[results/atomworks-corpus-validation.json](results/atomworks-corpus-validation.json),
and [INPUT_CONTRACT.md](INPUT_CONTRACT.md). The review has 100 recorded findings
with implemented fixes or explicit remaining limitations. Draft PR text is
prepared in [PR_DESCRIPTION.md](PR_DESCRIPTION.md); no PR has been published yet.

The user's latest instruction expands implementation: reuse as much current
AtomWorks code as practical and remove duplicated tmol code, with companion PRs
to AtomWorks `dev` if needed. A new isolated worktree
`/mnt/home/kdidi/projects/atomworks-dev-tmol-shared-input` starts from fetched
`origin/dev` at `59afb1e2`. The older tested companion branch `adcbfd76` and
original fixture checkout remain preserved. Consolidation and its final
validation/publication are still in progress; the optional reader is an
implemented migration step, not completion of the all-file/tensor unification.
