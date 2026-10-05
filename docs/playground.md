# Protein score playground

Rotate **Phe45 in ubiquitin** to relieve a side-chain clash. Find a lower total
score, and watch the individual contributions change. The rest of the protein
stays fixed.

```{raw} html
:file: _includes/playground.html
```

## What you are exploring

Each slider selects a conformation scored by TMol with `beta2016_score_function`.
There are **576 samples**, spaced 15° apart in two side-chain rotations relative
to the input geometry. All atoms, including hydrogens, enter the score; the viewer
shows heavy atoms. Scores are looked up exactly, with no energy interpolation.

The partner view shows the eight residues with the largest interaction changes
between the starting and best sampled states. Select one to locate it in the
structure. These pair contributions exclude Phe45's self terms.

The table shows weighted whole-protein contributions. “Change” compares each
term with the challenge's starting conformation. Improving one term can worsen
another. **Best sampled** chooses the lowest total on this grid; it does not
minimize the protein or run TMol's full packer.

Download the [scored dataset](_static/playground/phenylalanine.json) or regenerate
it with `python scripts/generate_score_playground.py` from the TMol checkout.
The dataset records the input hash, code revision, score weights, and sampling
settings. The input is ubiquitin [1UBQ](https://www.rcsb.org/structure/1UBQ).

Continue with {ref}`repacking <glossary-repacking>`,
{ref}`FastRelax <glossary-fastrelax>`, or the {doc}`quickstart` to run these
operations on your own structures.
