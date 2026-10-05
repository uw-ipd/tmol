# Score gaps: examples and reproduction

For Frank: these are **native tmol 0.1.58 vs PyRosetta 2024.39** comparisons on
identical benchmark inputs. The branch adds diagnostics only; production scoring
code is unchanged. Controls below are kept separate from the Figure 1 scores.

Validated end to end on 14 inputs: all eight audited cases reproduce the frozen
reference, and the eight-ligand connectivity result is reproduced exactly to
the reported precision. See `metadata/reproduction_validation.json`.

| Gap | Evidence | Interpretation |
| --- | --- | --- |
| Dunbrack deviation | COX1 Lys215: **663.095 vs 10.477 REU**. Restoring missing-well fallback entries gives **10.477**. 1KX5 C:Lys119: **344.660 → 7.090**, PyRosetta **7.090**. 1S78 A:Arg318: **102.044 → 1.645**, PyRosetta **1.645**. | Confirmed lookup defect: the LYS/ARG libraries lack aliases for 8/6 missing wells. The resolver leaves `-1`, which enters the scoring table offset. |
| Nucleic-acid solvation | 1KX5: tmol **5408.16**, PyRosetta **10645.67**; substituting only NA LJ/LK parameters gives **9831.35**, removing **84.45%** of the gap. | Different parameterization: tmol DNA types use `flags_opt46A.523`; this baseline uses `beta_nov16_cart`. Residual differences remain. |
| Ligand Cartesian bonded | Matching declared chain connectivity changes Pearson **−0.533 → 0.999994** across all eight successful ligand inputs. | Different gap handling. Retaining long declared bonds is a comparison control, not a physical repair. |
| Other terms | 6Q1H's remaining bonded discrepancy concentrates in PyRosetta improper torsions. LJ repulsion, proline and terminal Dunbrack residuals remain. Hydroxyl weights are **0.5 vs 1.0**. | Still unresolved; high pooled Pearson does not imply equal energies. |

Atom-alias matching finds effectively identical coordinates in eight audited
inputs (maximum difference 0.0000344 Å; two unmatched terminal atoms in 4LUP).
The large gaps are not explained by CPU-thread rounding.

## Run

Use an environment with tmol from this branch's base **`f92d40d12` / 0.1.58**,
and a separately installed, licensed **PyRosetta 2024.39**. The measured
Python/PyTorch/chemistry versions and wheel checksum are in `metadata/`.
No GPU, scheduler, network download or cluster-specific path is required for
these score controls. Ligand loading also needs tmol's ligand dependencies
(including Biotite, RDKit and AtomWorks). Use the repository's normal setup.

```bash
cd dev/score_gap_reproduction
python verify.py                         # archived input/evidence integrity
python run.py --datasets cox1            # clearest rotamer reproduction
python verify.py --results results      # compare with frozen native/control scores
python run.py --topology                 # eight audited inputs + eight-ligand control
```

If the engines use different environments, pass `--tmol-python /path/to/python`
and `--pyrosetta-python /path/to/python`. For an unpacked PyRosetta distribution,
set `PYROSETTA_PATH` to the directory containing `pyrosetta/`. `--output` selects
another result directory. Both engines use one software thread; each process
initializes its own native score function. Corrected future tmol versions may
intentionally differ from the frozen reference checks.

## Evidence and code pointers

- `reference/missing_well_outliers.csv`, `controlled_term_differences.csv` and
  `topology_diagnostic.csv`: numerical examples above. Compressed JSON records
  preserve coordinates, parameters, per-residue energies and all interventions.
- `scripts/control_dunbrack.py`: private-database fallback control, following
  [Rosetta's missing-well algorithm](https://github.com/RosettaCommons/rosetta/blob/main/source/src/core/pack/dunbrack/RotamericSingleResidueDunbrackLibrary.tmpl.hh).
  Inspect `tmol/score/dunbrack/_params.py::_create_rotind2tableinds` and
  `potentials/potentials.hh::{classify_rotamer_for_block,block_interpolate_rotameric_tables}`.
- `scripts/control_score_inputs.py` and `audit_ligand_topology.py`: coordinate,
  NA parameter and connectivity controls. tmol NA parameters are in
  `tmol/database/default/scoring/ljlk.yaml`.
- `figure/figure1.pdf`: updated FastRelax connecting lines; **68/87** timing
  configurations, including **18/27** PyRosetta inputs, complete at this snapshot.
  `python figure/plot_performance.py` regenerates performance panels and Pearson
  intervals from archived measurements (NumPy/Matplotlib; Helvetica optional).
  Remaining timings are pending, never replaced by historical measurements.
  `scripts/run_fastrelax.py --help` exposes the portable timing harness.

PyRosetta `cart_bonded_torsion` already includes `cart_bonded_improper`: do not
sum both. Interresidue energy allocation also differs between engines, so
single-residue bonded differences need care. Diagnostic score adjustments are
not incorporated into the native Pearson panel or FastRelax timings.
