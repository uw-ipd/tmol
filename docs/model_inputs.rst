Model coordinate inputs
=======================

Run `Model inputs <https://colab.research.google.com/github/uw-ipd/tmol/blob/ae1d35c6ecd243bdeba84d83ee3ce49e7254de81/notebooks/example_02_model_inputs.ipynb>`_
in Colab or open ``notebooks/example_02_model_inputs.ipynb`` locally for editable
adapters, saved predictions, and gradient checks.

Map model coordinates to named residues and atoms, then build TMol canonical
tensors. Keep model-specific keys and layouts in your application: TMol provides
``CanonicalOrdering`` / ``CanonicalForm``, but no longer provides OpenFold or
RoseTTAFold2 input adapters.

Declare a layout once
---------------------

Supply the ordered residue names corresponding to your integer token IDs and,
for each residue, the ordered atom names corresponding to its coordinate slots.
AtomWorks supplies reusable definitions such as ``AF2_ATOM14_ENCODING`` and
``AF2_ATOM37_ENCODING``. An encoding's atom names do not by themselves establish
your model's token numbering. Check both against the version that produced the
prediction. In particular, do not infer an encoding solely from 14 or 37 slots.

The notebook's ``prepare_named_layout`` function prepares these mappings once
and binds coordinates using Torch indexed assignment. It accepts batched
tensors, uses ``chain_id=-1``
for explicit residue padding, and distinguishes unobserved atoms from padded
residues. Finite supplied coordinates remain connected to autograd. Missing
coordinates are NaN triplets; infinity and partially finite triplets are rejected.

The adapter uses the default chemical database and rebuilds the pose on each
iteration; it caches only the source mapping. Supply all required heavy atoms.
The adapter does not pack missing side chains. Missing terminal atoms and
hydrogens use the
shared tmol construction code. For arbitrary ligands, PTMs or nucleic acids,
prepare a ``PoseBuildContext`` from annotated AtomArray topology first and use
its matching canonical ordering, packed types and connectivity.

OpenFold-style output
---------------------

The notebook's ``openfold_example`` extracts final atom14 coordinates, residue
tokens, an atom-existence mask, and chain IDs from a saved prediction. Adapt
its dictionary keys to your model's output schema.

A model producing atom37 can use the same named-layout function with its atom37
encoding and observation mask; no temporary atom14-to-atom37 expansion is needed.

RoseTTAFold2-style output, including supplied hydrogens
-------------------------------------------------------

Take ``num2aa`` and ``aa2long`` from the exact model checkout that produced
``seq`` and ``xyz``. Pass those tables as ``residue_names`` and ``atom_names``
to ``rf2_example``. RF2-family models place hydrogen slots at different offsets;
do not substitute the RF2AA/atom36 table for a legacy atom27 prediction.

Choose ``hydrogens="preserve"`` to retain supplied nonterminal hydrogen
coordinates, or ``hydrogens="rebuild"`` to mark them missing and rebuild them.
The generic amide ``H`` at each chain's N terminus is replaced by the appropriate
terminal hydrogen model in either case. The rebuilt input slots have no gradient;
other supplied coordinates retain their gradient connection.

Scoring and repeated guidance
-----------------------------

Keep the sequence, chain layout, atom mapping and chemical context fixed while
reusing a prepared mapping. Bind each new coordinate tensor without calling
``.numpy()``, detaching it, or writing a temporary structure file. Use separate
builders for different topology/sequence groups. TMol's score module can then
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
------------------------------------

The model-specific ordering and packed-type factories were removed together with
the input functions. Old serialized canonical integer tensors must be interpreted
with their original ordering, using the tmol version that wrote them, and exported
with named residue/atom identities before migration. Do not load their residue
indices against today's default ordering. The original prediction tensors used
by this tutorial remain independent of tmol's internal residue numbering.
