Build an interface for your model
================================

Run `Example 02: model inputs <https://colab.research.google.com/github/kierandidi/tmol/blob/review/pr503-chemistry-efficiency/notebooks/example_02_model_inputs.ipynb>`_
in Colab, or open ``notebooks/example_02_model_inputs.ipynb`` locally. It includes
installation, both editable adapters, saved predictions, score/gradient checks
and a repeated-guidance loop. The notebook is the executable source for this tutorial.


Model output is a coordinate tensor plus a description of what each slot means.
Keep the model's dictionary keys and version-specific layouts in your application.
Tmol provides the generic ``CanonicalOrdering`` / ``CanonicalForm`` construction
boundary; it no longer provides OpenFold or RoseTTAFold2 input interfaces.

Declare a layout once
--------------------

Supply the ordered residue names corresponding to your integer token IDs and,
for each residue, the ordered atom names corresponding to its coordinate slots.
AtomWorks supplies reusable definitions such as ``AF2_ATOM14_ENCODING`` and
``AF2_ATOM37_ENCODING``. An encoding's atom names do not by themselves establish
your model's token numbering. Check both against the version that produced the
prediction. In particular, do not infer an encoding solely from 14 or 37 slots.

The runnable example below prepares these mappings once and binds coordinates
using Torch indexed assignment. It accepts batched tensors, uses ``chain_id=-1``
for explicit residue padding, and distinguishes unobserved atoms from padded
residues. Finite supplied coordinates remain connected to autograd. Missing
coordinates are NaN triplets; infinity and partially finite triplets are rejected.

The notebook defines ``prepare_named_layout`` in an editable code cell.

This small example uses the default chemical database and calls canonical pose
construction on every iteration. It caches the source mapping, not the final
pose topology. Supply all required heavy atoms; missing sidechain packing is
not implemented by this example. Missing terminal atoms and hydrogens use the
shared tmol construction code. For arbitrary ligands, PTMs or nucleic acids,
prepare a ``PoseBuildContext`` from annotated AtomArray topology first and use
its matching canonical ordering, packed types and connectivity.

OpenFold-style output
---------------------

This example extracts the final structure-module atom14 coordinates, residue
tokens, atom-existence mask and chain IDs from a saved prediction. These dictionary
keys describe the example prediction, not a universal OpenFold output schema.
If your model returns a different schema, adapt this extraction locally.

The notebook defines ``openfold_example`` in an editable code cell.

A model producing atom37 can use the same named-layout function with its atom37
encoding and observation mask; no temporary atom14-to-atom37 expansion is needed.

RoseTTAFold2-style output, including supplied hydrogens
-----------------------------------------------------

Take ``num2aa`` and ``aa2long`` from the exact model checkout that produced
``seq`` and ``xyz``. Pass those tables as ``residue_names`` and ``atom_names``
below. Different RF2-family models place hydrogen slots at different offsets;
do not substitute the RF2AA/atom36 table for a legacy atom27 prediction.

Choose ``hydrogens="preserve"`` to retain supplied nonterminal hydrogen
coordinates, or ``hydrogens="rebuild"`` to mark them missing and rebuild them.
The generic amide ``H`` at each chain's N terminus is replaced by the appropriate
terminal hydrogen model in either case. The rebuilt input slots have no gradient;
other supplied coordinates retain their gradient connection.

The notebook defines ``rf2_example`` in an editable code cell.

Scoring and repeated guidance
-----------------------------

Keep the sequence, chain layout, atom mapping and chemical context fixed while
reusing a prepared mapping. Bind each new coordinate tensor without calling
``.numpy()``, detaching it, or writing a temporary structure file. Use separate
builders for different topology/sequence groups. Tmol's score module can then
backpropagate into the supplied coordinates::

    pose = builder(coords, observed=mask)
    module = score_function.render_whole_pose_scoring_module(pose)
    energy = module(pose.coords).sum()
    gradient, = torch.autograd.grad(energy, coords)

Discrete chemical preparation and rotamer selection are not differentiable.
An optimized prepared-topology builder for repeated guidance is a separate layer
from this tutorial's minimal adapter. AtomWorks' tensor conversion utilities
retain chemical identities with NaN coordinates; they do not generally impute
finite missing coordinates or preserve a Torch tape through NumPy conversion.

Migration of saved canonical tensors
-----------------------------------

The model-specific ordering and packed-type factories were removed together with
the input functions. Old serialized canonical integer tensors must be interpreted
with their original ordering, using the tmol version that wrote them, and exported
with named residue/atom identities before migration. Do not load their residue
indices against today's default ordering. The original prediction tensors used
by this tutorial remain independent of tmol's internal residue numbering.
