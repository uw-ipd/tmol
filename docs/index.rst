.. TMol documentation master file, created by
   sphinx-quickstart on Sun Jul 15 11:02:35 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/uw-ipd/tmol

TMol Documentation
================================

TMol is a tensor-based molecular modeling library for use on GPUs and CPUs.

Generalized preparation of noncanonical polymers, ligands and covalent
attachments is described in :ref:`noncanonical-chemistry`. Start with
:ref:`chemistry-workflows` for scoring, packing and relaxation recipes, and
:ref:`rosetta-comparison` for inherited methods, implementation differences
and validation boundaries. These guides identify the feature revision they
describe; they do not imply that an unmerged feature is in every release.

.. The toctree entry declares the document's location, and children, inside the
  table-of-contents entry. We want two captioned toc constructs on the front,
  "Notes", containing freeform documentation, and "Packages", docstring based
  inline documentation. We declare a single root (non-:name:) toc entry that
  contains our notes + apidoc, then create the `apidoc` toc entry inline in this
  index, allowing both to show up on navigation sidebar. If we, instead, declared 
  two root toc entries they would both show up in *this* document's sidebar, but
  one would be lost when navigating to subpages.

.. toctree::
  :glob:
  :caption: Notes

  architecture
  noncanonical_chemistry
  chemistry_workflows
  rosetta_comparison
  datatypes
  apidoc

Packages
--------

.. Generate sidebar entries via a hidden toctree entry, then inline the top-level
   packages as an autosummary table so there's a touch of inline documentation.
   Make sure to update *both* tables when adding top-level components.

.. toctree::
  :caption: Packages
  :name: apidoc
  :hidden:

  apidoc/tmol.pose
  apidoc/tmol.chemical
  apidoc/tmol.ligand
  apidoc/tmol.pack
  apidoc/tmol.optimization
  apidoc/tmol.relax
  apidoc/tmol.score
  apidoc/tmol.database
  apidoc/tmol.kinematics
  apidoc/tmol.numeric
  apidoc/tmol.types
  apidoc/tmol.utility
  apidoc/tmol.io
  apidoc/tmol.support
  apidoc/tmol.extern

.. autosummary::

  tmol.pose
  tmol.chemical
  tmol.ligand
  tmol.pack
  tmol.optimization
  tmol.relax
  tmol.score
  tmol.database
  tmol.kinematics
  tmol.numeric
  tmol.types
  tmol.utility
  tmol.io
  tmol.support
  tmol.extern

Indices
==================

* :ref:`genindex`
* :ref:`modindex`
