Input and Output
================

The public :mod:`tmol.io` API converts common structure representations to and
from :class:`tmol.pose.PoseStack`.

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
by default. Generated ligand types may rebuild hydrogens unless
``trust_hydrogen_names=True``.

Use :func:`tmol.io.atom_array_from_file` or
:func:`tmol.io.pose_stack_from_file` for PDB, CIF, compressed CIF and binary CIF.
The reader options are ``model``, ``assembly_id`` and ``use_ccd``. Reading preserves
supplied names, coordinates, hydrogens and covalent bonds; declared unresolved
heavy atoms carry NaN coordinates. ``use_ccd=False`` disables dictionary lookup,
including for PDB and custom residue names. Missing chemistry is allowed at this
stage.

Known residues need only names and coordinates. Pass an existing ``param_db`` or
load prepared ``ligand_params_files`` to reuse ligand parameters with a coordinate-only
PDB/CIF. With ``prepare_ligands=True``, pose preparation generates only missing
parameters and reports the residue name when required bond orders are absent.
Use ``return_context=True`` to retain the resulting parameter database for scoring
and repeated construction. Direct AtomArray inputs follow the same contract.

.. code-block:: python

   # Reuse known chemistry; the PDB needs only matching names and coordinates.
   pose = pose_stack_from_file(
       "complex.pdb", device, use_ccd=False,
       ligand_params_files=["ligand.tmol"],
   )
   # Prepare missing chemistry once, then reuse context.parameter_database.
   pose, context = pose_stack_from_file(
       "complex.cif", device, prepare_ligands=True, return_context=True,
   )

:func:`tmol.io.pose_stack_from_pdb` accepts paths, text and lists of lines through
the same file path. File and AtomArray inputs preserve matching supplied coordinates
for ordinary protein residues. Pose construction builds missing hydrogens but
preserves finite authored protein hydrogen coordinates by default. Pass
``no_optH=False`` to request OptH packing. Generated ligand types may rebuild
hydrogens whose names changed during parameter generation; pass
``trust_hydrogen_names=True`` only when those names match the prepared database.
Low-level PDB atom-record/DataFrame utilities retain their canonical-only contract.

.. automodule:: tmol.io
   :members:
   :imported-members:
   :show-inheritance:
