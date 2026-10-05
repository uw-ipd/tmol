# Extending TMol

Choose an API for the property you need to change.

## Extension APIs

| Change | API | Requirement |
| --- | --- | --- |
| Restrict a known chemical alphabet | `ParameterDatabase.create_stable_subset()` | The subset must still contain every input block and required variant. |
| Add a ligand or residue definition | Derive a new `ParameterDatabase` through the ligand or residue-injection API | Keep the process default immutable and build the score function from the derived database. |
| Change the contribution of an existing score lane | `ScoreFunction.set_weight()` | A weight change does not change the underlying parameters. |
| Change an energy model parameter | Derive the relevant immutable scoring database | Validate values and gradients independently; the result is a new model. |
| Change coordinate dependencies | Construct an explicit `FoldForest` | A FoldForest is not chemical connectivity, docking, or idealization. |
| Restrict initial design identities | Subclass `PackerPalette` | Start from compatible choices and only remove entries. |
| Restrict positions or candidates for one job | Modify a `PackerTask` monotonically | A task cannot safely re-enable choices excluded earlier. |

## Validate an extension

1. Build the smallest system that contains the changed chemistry or topology.
2. Compare a derived object with its unchanged parent rather than mutating
   global state.
3. Inspect candidate sets before running stochastic search.
4. Assert which identities, coordinates, or score lanes must remain unchanged.
5. Test coordinate gradients when the extension participates in optimization.
6. Repeat the complete scientific workflow on CPU and CUDA where both are
   supported.

These checks establish implementation behavior. New parameters still need
validation against suitable molecular data.

## Examples and reference

{doc}`Chemistry and scoring </tutorial/11_extending_chemistry_and_scoring>` · {doc}`Kinematics </tutorial/12_explicit_foldforests_and_torsions>` · {doc}`Packer extensions </tutorial/13_extending_the_packer>`
