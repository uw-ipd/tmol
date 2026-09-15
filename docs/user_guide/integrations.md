# Model and structure integrations

TMol is designed to sit inside PyTorch-based structural-biology workflows. It can
score structures loaded from standard files and convert outputs from structure
prediction systems into `PoseStack` objects.

> - **Prerequisites:** {doc}`Quickstart </quickstart>` and the output schema for
>   the source model or structure library.
> - **Deep tutorial:** {doc}`01 — Working with TMol
>   </tutorial/01_working_with_tmol>`.
> - **Related workflows:** {doc}`GPU batching </workflows/gpu_batching>` and
>   {doc}`Ligand preparation </user_guide/ligands>`.
> - **API reference:** {doc}`Input and Output </api/io>` and
>   {doc}`Pose </api/pose>`.
> - **Rosetta mapping:** {doc}`I/O, selections, and options
>   </tutorial/rosetta_crosswalk>`.

## Structure files and annotated arrays

AtomWorks handles PDB, CIF, compressed CIF and binary CIF parsing. The shared
constructor retains chemical identity and declared bonds, including unresolved
atoms at NaN:

```python
import torch

from tmol.io import pose_stack_from_file
from tmol.score import beta2016_score_function

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pose_stack, context = pose_stack_from_file(
    "complex.cif",
    device,
    prepare_ligands=True,
    return_context=True,
)
sfxn = beta2016_score_function(device, param_db=context.parameter_database)
```

Use `atom_array_from_file()` to inspect or select the parsed topology before
passing it to `pose_stack_from_biotite()`. An existing AtomWorks or Biotite array
uses that same constructor. Preserve its bond table and chemical annotations
when selecting atoms. `pose_stack_from_pdb()` retains the PDB filename/lines,
residue-slicing and supplied-hydrogen compatibility policy.

Parsing marks unresolved atoms; TMol reconstructs supported missing coordinates
from chemical construction frames. An unresolved backbone or an underdetermined
ligand requires an explicit input policy or additional geometry. The score
function must use the pose's device and prepared parameter database. See
{doc}`07 — Ligands and Parameter Files </tutorial/07_ligand_and_params>` for
preparation, strict errors and reusable parameter bundles.

## Direct prediction tensors

Canonical proteins in AtomWorks' `UNIFIED_ATOM37_ENCODING` can use
`pose_stack_from_canonical_aa_atom37(coords, residue_type, chain_iid)` directly. Coordinates
have shape `[batch, residues, 37, 3]`; token IDs and chain IDs have shape
`[batch, residues]`. The function uses AtomWorks' token order and atom names.
Tensor shape alone does not identify an encoding.

For atom14 or another named layout, map the predictor's token IDs and atom
names into `CanonicalForm`, then call `pose_stack_from_canonical_form()`. The
{download}`executable model-input tutorial <../../notebooks/example_02_model_inputs.ipynb>`
implements OpenFold and RoseTTAFold2 examples using this shared contract. It
shows final-recycle selection, explicit masks, batched scoring and gradients,
and both preservation and rebuilding of RF2's supplied hydrogen slots. Those
model-specific mappings belong in the caller; TMol has no separate OpenFold or
RF2 constructor.

Keep coordinate mapping in Torch to preserve gradients to prediction tensors.
When inference globally disables gradients, use `torch.enable_grad()` around
construction and scoring. Differentiate the score through `pose_stack.coords`
without detaching it. Cache the named-layout mapping for repeated predictions;
use the prepared topology path below for repeated Atom37 guidance or search.

## Differentiable AtomWorks Atom37 coordinates

For a model that predicts AtomWorks unified Atom37 coordinates, keep chemical
identity and connectivity in its Biotite `AtomArray` and route only coordinates
from the model tensor. Annotate each supported atom with its model `token_id`
and `atom37_slot`, build the chemistry context once, and reuse it across model
steps:

```python
import torch

from tmol.io import (
    build_context_from_biotite,
    prepare_atom37_pose_builder,
)
from tmol.score import beta2016_score_function

# atom_array is the AtomWorks-produced topology. atom37_slots is a per-atom
# integer array from the same unified encoding; use -1 for an unmapped atom.
atom_array.set_annotation("atom37_slot", atom37_slots)

context = build_context_from_biotite(
    atom_array,
    atom37_coords.device,
    prepare_ligands=True,
)
pose_builder = prepare_atom37_pose_builder(atom_array, context)

# One call accepts one sample or a same-topology batch. The builder can be
# reused at every diffusion, guidance, or search step.
pose_stack = pose_builder(atom37_coords)  # [sample, token, 37, xyz], float32
sfxn = beta2016_score_function(
    pose_stack.device,
    param_db=context.parameter_database,
)
scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
score = scorer(pose_stack.coords).sum()
score.backward()
```

A single call handles one sample or a whole same-topology batch; hydrogen
optimization is enabled by default. For high-throughput diffusion or search,
reuse both `pose_builder` and the scoring module for every compatible topology
and batch shape. Pass `opt_h=False` to the builder only when ideal hydrogen
placement is an intentional speed/accuracy tradeoff. For one-off conversion,
`pose_stack_from_atom37_and_topology(atom37_coords, atom_array, context)` remains
available.

A single `AtomArray` may provide topology for a batch of coordinate tensors. An
`AtomArrayStack` must either have the same number of models as the tensor batch
or one model that can be broadcast. Finite mapped tensor coordinates replace
the reference coordinates; unmapped slots (`-1`) and non-finite tensor entries
retain their reference coordinates. TMol-generated atoms, such
as hydrogens, are left in their built or optimized positions. Gradients from
TMol coordinates route back to the mapped Atom37 entries even when hydrogen
optimization is enabled. Missing, out-of-range, or ambiguous routing annotations
raise `Atom37MappingError`, so callers can handle mapping failures without
catching unrelated pose-construction errors.

The context defines supported chemistry: canonical proteins, nucleic acids,
prepared noncanonical polymers, ligands, glycans and fragmented covalent groups
all use this interface. Metal parameterization remains separate. The adapter
has no residue or element allowlist; strict preparation reports chemistry that
the context cannot represent. A prepared builder owns mutable caches: use one
per calling thread, and rebuild it when chemical identity or connectivity changes.
