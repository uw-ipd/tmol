Tutorials
=========

A single notebook walks through TMol: loading structures, scoring and
optimization, metals and nucleic acids, ligands and NCAAs, selections,
constraints, and advanced kinematics and topics. Each section works through
its operations with code, results, and structure viewers side by side, and
can be run independently once the shared Setup cell at the top has run.
Sections marked **TODO** are still being written.

Setup starts folded. Expand it when you need the details; the modeling steps and
their results stay visible.

In Colab, select a GPU under **Runtime → Change runtime type**, then **Run all**.
The shared ``notebook_setup`` module installs a tested TMol release with
operators matching Colab's PyTorch CUDA major and downloads the example data.
For local notebooks and supported environments, follow :doc:`installation`.

.. nbgallery::

   tutorial/01_tmol_tutorial
