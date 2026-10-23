Input and Output
================

The public :mod:`tmol.io` API converts common structure representations to and
from :class:`tmol.PoseStack`.

Once the residue types are resolved, :func:`tmol.io.pose_stack_from_atom37`
builds a pose from tensors alone, for canonical and noncanonical chemistry
alike: residue identity comes from ``res_types`` and any further covalent
chemistry from ``covalent_bonds``. Its slot layout is built by
:func:`tmol.io.atom37_slot_map_for_ordering`. The direct AtomWorks adapter is
the canonical-amino-acid special case of that path. To read chemistry from a
structure instead, :func:`tmol.io.pose_stack_from_atom37_and_topology` derives
the same information from general Biotite topology, including nucleic acids and
ligands. Repeated diffusion,
guidance, and search workloads should bind their fixed topology once with
:func:`tmol.io.prepare_atom37_pose_builder`; its returned callable accepts
each coordinate batch and preserves ordinary finite authored protein hydrogens
by default. Generated ligand types rebuild their hydrogen coordinates.

Use :func:`tmol.atom_array_from_file` or
:func:`tmol.pose_stack_from_file` for PDB, CIF, compressed CIF and binary CIF.
The reader options are ``model`` and ``assembly_id``. Reading preserves
supplied names, coordinates, hydrogens and covalent bonds; declared unresolved
heavy atoms carry NaN coordinates. Chemistry the file leaves out is completed
from the component dictionary. Missing chemistry is allowed at this stage.

Known residues need only names and coordinates. Pass an existing ``param_db`` or
load prepared ``ligand_params_files`` to reuse ligand parameters with a coordinate-only
PDB/CIF. With ``prepare_ligands=True``, pose preparation generates only missing
parameters and reports the residue name when required bond orders are absent.
Use ``return_context=True`` to retain the resulting parameter database for scoring
and repeated construction. Direct AtomArray inputs follow the same contract.

.. code-block:: python

   # Reuse known chemistry; the PDB needs only matching names and coordinates.
   pose = pose_stack_from_file(
       "complex.pdb", device, ligand_params_files=["ligand.tmol"],
   )
   # Prepare missing chemistry once, then reuse context.parameter_database.
   pose, context = pose_stack_from_file(
       "complex.cif", device, prepare_ligands=True, return_context=True,
   )

:func:`tmol.pose_stack_from_file` also accepts MOL2 and SDF, file text and
streams. File and AtomArray inputs preserve matching supplied heavy-atom
coordinates for ordinary protein residues and build missing hydrogens. By
default, hydrogens are packed with OptH; pass ``no_optH=True`` to preserve finite
authored protein hydrogen coordinates instead. ``ligand_hydrogens`` controls the
hydrogens of generated ligand types: ``"input"`` keeps the input names and the
coordinates of hydrogens that sit at bond length from the parent their name
gives them, ``"names"`` (the default) keeps the names but rebuilds coordinates,
and ``"rebuild"`` generates names and rebuilds coordinates.

Write with :func:`tmol.file_from_pose_stack` or
:func:`tmol.file_from_atom_array`, which produce PDB or CIF and return the
text when no path is given.

.. automodule:: tmol.io
   :members:
   :imported-members:
   :show-inheritance:
