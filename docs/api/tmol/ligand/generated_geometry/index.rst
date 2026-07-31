tmol.ligand.generated_geometry
==============================

.. py:module:: tmol.ligand.generated_geometry

.. rubric:: Module docstring

.. code-block:: text

   Corrections to the generated (OpenBabel) 3D structure.
   
   In some cases, openbabel-generated conformers are incorrect in known,
   predicible ways.  This module handles those predictions, correcting
   very specific geometry issues before generating parameters.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.generated_geometry.logger


Functions
---------

.. autoapisummary::

   tmol.ligand.generated_geometry.planarize_conjugated_nh2
   tmol.ligand.generated_geometry.correct_generated_geometry


Module Contents
---------------

.. py:data:: logger

.. py:function:: planarize_conjugated_nh2(mol: rdkit.Chem.Mol) -> list[str]

   .. rubric:: Docstring

   .. code-block:: text

      Make -NH2 groups conjugated to an sp2 center planar.
      
      Correct proton geomoetry around amine N bound to sp2 heavy atoms.
      Both hydrogens are rebuilt in the neighbor's plane, 120 deg off the C-N
      bond, preserving each N-H bond length and each hydrogen's original side so
      the correction is minimal.
      

.. py:function:: correct_generated_geometry(mol: rdkit.Chem.Mol) -> list[str]

   .. rubric:: Docstring

   .. code-block:: text

      Repair known defects in the generated conformer, in place.
      
      Apply all corrections specified in _CORRECTIONS.
      

