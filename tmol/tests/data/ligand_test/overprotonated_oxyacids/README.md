# Over-protonated oxyacid fixtures

Mol2 files whose acid groups carry more protons than any real acid holds, as an
external preprocessing step that protonates every terminal oxygen produces them.
Each molecule comes in two forms:

| suffix | what was done to the neutral acid |
|--------|-----------------------------------|
| `_all_single` | every center-O bond single, every terminal O protonated: `[P+](O)(O)O`, `S(O)(O)O`, `C(O)O` |
| `_oxonium` | bond orders kept, H added to the double-bonded oxygens: `P(=[OH+])`, `C(=[OH+])` |

| file stem | neutral acid | charge at pH 7.4 |
|-----------|--------------|------------------|
| `methyl_phosphate` | `COP(=O)(O)O` | -2 |
| `methyl_sulfate` | `COS(=O)(=O)O` | -1 |
| `acetic_acid` | `CC(=O)O` | -1 |

Built rather than taken from a real input: RDKit embedded the neutral acid
(MMFF, seed 11), then the oxygens were protonated with H placed 1.0 A from
the oxygen, and the files were written with `NO_CHARGES` and Tripos `X.3` types.
