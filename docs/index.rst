.. TMol documentation master file, created by
   sphinx-quickstart on Sun Jul 15 11:02:35 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/uw-ipd/tmol

TMol Documentation
================================

TMol is a tensor-based molecular modeling library for use on GPUs and CPUs.

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

  apidoc/tmol.system
  apidoc/tmol.score
  apidoc/tmol.database
  apidoc/tmol.kinematics
  apidoc/tmol.numeric
  apidoc/tmol.types
  apidoc/tmol.utility
  apidoc/tmol.io
  apidoc/tmol.viewer
  apidoc/tmol.support
  apidoc/tmol.extern

.. autosummary::

  tmol.system
  tmol.score
  tmol.database
  tmol.kinematics
  tmol.numeric
  tmol.types
  tmol.utility
  tmol.io
  tmol.viewer
  tmol.support
  tmol.extern

Indicies
==================

* :ref:`genindex`
* :ref:`modindex`
