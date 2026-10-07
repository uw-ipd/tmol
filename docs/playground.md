# Protein score playground

Rotate Phe45 in ubiquitin and see how the score changes. Orange marks the
moving side chain; all other atoms stay fixed. Lower scores are favored.

```{raw} html
:file: _includes/playground.html
```

## Data and method

The sliders select **576 conformations**, scored with `beta2016_score_function`
at 15° intervals relative to the input geometry. TMol scores all atoms,
including hydrogens; the viewer shows heavy atoms. The browser reads these
recorded values without interpolation. **Best sampled** selects the lowest
score on this grid.

The term table reports weighted whole-protein scores. Residue interactions
exclude Phe45's self terms and show the eight partners with the largest changes
between the starting and best sampled states. All changes are relative to the
starting conformation.

[Download the dataset](_static/playground/phenylalanine.json), including input
hashes, code revision, and scoring settings. To regenerate it from the bundled
[1UBQ](https://www.rcsb.org/structure/1UBQ) structure:

```bash
python scripts/generate_score_playground.py
```

For a full side-chain search, see {ref}`repacking <glossary-repacking>`.
The {doc}`quickstart` shows how to score, repack, and relax your own structures.
