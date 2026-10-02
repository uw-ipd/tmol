# Carbon-bound terminal hydrogen: minimal graph-preservation reproducer

`3t0x_B106_prepared.pdb` contains 17 original atom rows and the two original
C–HXT `CONECT` rows for prepared 3T0X VAL B106. No atom coordinates, names,
charge fields or retained bond declarations were edited. The only operation
was cropping to this residue and appending `END`. These rows are also in the
[public tmol regression fixture](https://github.com/uw-ipd/tmol/blob/0d834c2a362d6882a0cedf61c1771823ef8ab68f/tmol/tests/data/sweep_regressions/terminal_aldehyde_3t0x.pdb).

Run in an environment with tmol and its public AtomWorks dependency installed:

```sh
ALLOW_BIOTITE_CCD=1 python reproduce.py
```

The script reads source PDB atoms and bonds directly with Biotite, calls the
public `pose_stack_from_biotite(..., no_optH=True)` API, and verifies that the
caller input is unchanged. Direct source reading avoids mistaking the main
file reader's NaN missing-atom placeholders for deposited atoms.

Expected current result: an explicit unsupported-chemistry `ValueError` naming
`B:106:VAL/C`, with one supplied hydrogen versus zero supported by the database.

For an already built alternate tmol checkout or export:

```sh
ALLOW_BIOTITE_CCD=1 python reproduce.py --source-root /path/to/tmol-checkout --expect silent-conversion --output result.json
```

Verified source revisions:

- Historical source tree: `6a4ec864465bfabf4082ae045eef0c25f73c7abb`, public
  commit `c15881279316a0d1b4613838e2071a38f34443d8` (byte-identical to the tested
  scoped commit `2de72eb6c700810bd4b698a1829b148f3eb52236`). It returns
  `VAL:cterm:nterm`, removes HXT from the pose and adds OXT.
- Current tmol: `0d834c2a362d6882a0cedf61c1771823ef8ab68f`; rejects clearly.
- Both runs use public AtomWorks `e0f2e4da0aef2cf6c81a2c506eece3bc4204b75d`.
- Python 3.12, PyTorch 2.14.1, CPU. Existing compiled extensions were reused
  because native source is unchanged between these two tmol trees.

The cropping also makes the N terminus free; the old converter consequently
adds normal N-terminal hydrogen atoms. The demonstrated defect concerns the
unchanged, explicitly supplied **C–H** edge at the opposite terminus.

## Why this is probably a prepared-model cap, not an experimental aldehyde

The frozen January 6, 2026 [deposited 3T0X mmCIF](https://files.rcsb.org/download/3T0X.cif.gz)
contains VAL B106 as polymer position 109 of 123. LEU B106A (position 110),
ASP B107 and subsequent residues are explicitly listed as unobserved. Thus
B106 is the last *resolved* residue, not the declared protein sequence end.
Its deposited heavy coordinates agree exactly with the prepared file. The
carbon-bound HXT appears only in the prepared file, whose header says it was
written by Maestro.

This supports a chain-truncation/capping artifact. It does not establish the
specific preparation command, nor prove that experimental aldehydes are
impossible. Absence of deposited hydrogen alone would not be enough evidence.

## Chemistry and appropriate fixes

The supplied graph locally says `CA–C(=O)–H`: a valence-valid aldehyde-like cap.
The raw PDB charge fields are blank. CONECT explicitly identifies the C–H
edge but does not encode bond order; a one-neighbor hydrogen implies a single
bond. HXT's conventional CCD association with OXT must not override that
explicit source edge.

Silently choosing `CA–C(=O)–OXT` changes the molecule, even if a carboxylate was
what the preparer intended. The current error is the safe behavior when the
force field has no matching form.

Two distinct remedies should stay explicit:

1. If the input was accidentally capped during preparation, repair/regenerate
   that prepared structure using original polymer sequence, missing-residue
   records and a declared gap-capping policy. An unresolved continuation does
   not determine a unique replacement atom or cap from the local graph.
2. For intentional aldehydes, add a residue/terminal form retaining carbon H,
   valid types/partial charges and complete bonded/nonbonded parameters.
   An explicit custom-component prototype can reuse CDp/Oal/HC/Nbb types, but
   requires special registration and routing and still rebuilds supplied H;
   ordinary automatic canonical-residue loading remains unsupported.
   Validate finite energies, analytic-versus-numerical gradients and bond
   preservation during minimization. A placement template alone is insufficient.
