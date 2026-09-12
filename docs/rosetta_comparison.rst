.. _rosetta-comparison:

Relationship to Rosetta
=======================

tmol inherits substantial scientific foundations from Rosetta: energy-function
families, atom types and parameters, rotamer libraries, internal-coordinate
modeling, packing and pack/minimize relaxation. Its contribution is to
reorganize and extend these ideas around batched tensors, explicit chemical
metadata, shared setup and PyTorch coordinate differentiation. Functional
ancestry, implementation differences and scientific validation are different
claims.

This comparison uses tmol PR 503 commit
``c03c1e745f3bc655948ea12dac44d6c74620358f`` and Rosetta commit
``de92a3c0dea8a010d372a22025e3e50bd4e2f33f``. Source links are pinned to these
revisions. Neither a newer tmol branch nor an old Rosetta tutorial alone
establishes a feature's absence from current Rosetta.

What is unified
---------------

.. list-table:: Implementation comparison
   :header-rows: 1
   :widths: 20 39 41

   * - Concept
     - Rosetta already provides
     - tmol implementation and extension
   * - Chemical vocabulary
     - ResidueType, residue sets, patches, connections and caches
     - Prepared blocks and a shared batch-local PackedBlockTypes table
   * - Noncanonical setup
     - Params, polymer tools, MakeRotLib, ROTAMER_AA and BACKBONE_AA
     - Profile-driven preparation and explicit references in one database/context path
   * - Mixed bonded scoring
     - GenericBondedPotential with hybrid exclusions
     - Packed type hierarchies and shared intra-/inter-block torsion gates
   * - D amino acids
     - Mirrored residues and scoring/sampling support
     - Mirrored tables enter the same packed table and kernel paths as L types
   * - Glycans and covalent links
     - GlycanSampler/GlycanTreeModeler, linkage conformers and chemical FoldTree edges
     - Generic anchor-plus-attachment conformers collapsed into one packer choice
   * - Ligand fragments
     - molfile_to_params can split ligands and create connections
     - Fragment annotations, propagated parameters and identity-restoring export
   * - Scoring and derivatives
     - Shared EnergyMethods, analytical derivatives and caches
     - Rendered tensor modules and coordinate derivatives in PyTorch autograd

Rosetta's `Residue params reader`_ documents ``ROTAMER_AA``, ``BACKBONE_AA``
and explicit Ramachandran-map settings. `Rosetta residue cache`_ shows shared
and pose-local residue-type caching. Explicit metadata, reuse and support for
noncanonical chemistry are not inventions of tmol. The practical advance is
making preparation and execution more uniform within its batched differentiable
framework. See also `Rosetta NCAA guide`_.

Hybrid bonded scoring: explicit ownership
-----------------------------------------

At a protein-backbone/generic-sidechain boundary, parameter lookup must span
both type vocabularies. tmol expands the generic hierarchy with Rosetta-type
fallbacks. Proper torsions whose two central atoms are both Rosetta-typed are
excluded from generic scoring; nucleic-acid torsion bonds and named omega
connections are separately gated to avoid competing contributions. Generic
impropers and Cartesian planarity terms use complementary typing logic.
Intra-block resolution occurs during setup; inter-block lookup uses packed
hierarchy and bond-type data. See `Generic torsions`_,
`Generic parameter database`_, and `Cartesian bonded`_.

This implements a Rosetta-inspired hybrid partition; it does not reproduce
every exclusion rule in `Rosetta generic bonded`_. Rosetta also considers
reference residues, changed atoms, backbone membership and the availability of
rotamer/Ramachandran information. tmol keeps bond lengths and angles in its
Cartesian bonded path while its generic term handles proper and improper
torsions. Term names therefore need not define identical partitions. Compare
matched physical contributions and parameter choices rather than assuming
name-for-name numerical identity.

The partition is intended to avoid double counting. It is not proof that each
torsion has a valid restraint: unresolved generic parameter lookups can be
omitted. Four-backbone-class scoring tests and finite-difference checks exercise
important mixed boundaries; they are not a complete parameter-coverage proof.
See `Noncanonical scoring tests`_ and `Generic gradient tests`_.

Proposal distributions and scoring potentials
---------------------------------------------

The noncanonical amino-acid sampler can borrow a canonical library by chemical
correspondence, while Dunbrack scoring only uses an actual library for the
residue's base name. Generic torsions supply novel sidechain torsional energy.
This allows useful proposals without treating the canonical sidechain's
frequency as a measured probability for the modification. An alpha-backbone
Ramachandran reference can still be borrowed (with an alanine fallback), and a
nucleotide's base reference can influence its torsion energy. Disclose and
validate these approximations rather than calling them newly fitted statistics.
See `Polymer builder`_, `Dunbrack scoring`_ and `NA sampler`_.

For native D residues, tmol reflects backbone table coordinates, reverses chi
means and well labels, preserves appropriate unsigned quantities, and packs
the resulting libraries through the normal machinery. This makes handedness
explicit across preparation, scoring and sampling. Rosetta already has
`Rosetta D amino acid support`_. tmol's mirror-image test uses symmetrized
glycine tables and disables hydrogen optimization to preserve its controlled
comparison; this is not an unconditional symmetry claim for every default
protocol. See `Mirrored libraries`_ and `Mirror tests`_.

Joint group packing and chemical kinematics
-------------------------------------------

The group representation changes the packer's decision variable. If blocks
:math:`a,b,c` belong to one sampled group, it may select
:math:`(a_k,b_k,c_k)` but not :math:`(a_k,b_l,c_m)` with unrelated indices.
Within-group contributions map to the representative's self-energy;
interactions with other blocks map to representative-partner energies. Sparse
coalescing accumulates all contributions, and selected coordinates are written
back to all members. Scoring remains decomposed by chemical block. This exposes
attached-group conformations to the ordinary discrete optimizer without
allowing inconsistent independent choices. See `Group packing`_.

Rosetta's `Rosetta glycan sampler`_ composes linkage sampling, glycan branch
minimization and packing; `GlycanTreeModeler paper`_ describes a validated,
specialized glycan protocol. tmol's generic group-choice representation is a
different integration strategy. It does not show that Rosetta cannot move
bonded groups or that tmol replaces specialized glycan potentials and search
protocols. Group sampling also differs from user-defined fragmentation, which
subdivides one prepared ligand for representation and interaction analysis.
Rosetta has `Rosetta ligand fragmentation`_ too.

Rosetta's ``Edge::CHEMICAL`` also predates tmol's chemical FoldForest edge.
tmol integrates this edge with tensor kinematics and derivative propagation.
A spanning forest cannot retain every ring edge as a tree edge. Chemical
closure remains in scoring, but exact closure under arbitrary
internal-coordinate changes requires more than this representation. Rosetta
provides `Rosetta GeneralizedKIC`_ for analytical closure of diverse covalently
connected chains.

What has to be measured
-----------------------

The new branch corrects Cartesian peptide-planarity records and generic type
lookup, so old tmol energy goldens and speed/parity results cannot simply be
reused. Report exact revisions, prepared parameters, term/weight alignment,
charge model, termini, protonation, device and precision. Rosetta-derived
functional forms do not establish agreement for automatically generated
parameters. A within-tmol golden is a regression check, not an independent
Rosetta comparison.

Separate setup time, repeated scoring/backward time, packing time and
end-to-end relaxation. Compare matched hardware budgets, candidate counts and
output quality. Report every chemical-coverage attempt and distinguish
construction, actual sampling, parameter coverage, finite derivatives,
geometric validity and scientific accuracy. Keep unsupported chemistry and
failed runs in the denominator.

Source evidence supports integration and broader automated preparation within
tmol. PDB-wide coverage, better physical accuracy, complete Rosetta replacement
or universally higher speed require experiments.

.. include:: _source_links.inc
