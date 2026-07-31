tmol
====

.. py:module:: tmol


Submodules
----------

.. toctree::
   :maxdepth: 1

   /api/tmol/chemical/index
   /api/tmol/database/index
   /api/tmol/io/index
   /api/tmol/kinematics/index
   /api/tmol/ligand/index
   /api/tmol/numeric/index
   /api/tmol/optimization/index
   /api/tmol/pack/index
   /api/tmol/pose/index
   /api/tmol/score/index
   /api/tmol/support/index
   /api/tmol/types/index
   /api/tmol/utility/index


Functions
---------

.. autoapisummary::

   tmol.include_paths


Package Contents
----------------

.. py:function:: include_paths()

   .. rubric:: Docstring

   .. code-block:: text

      C++/CUDA include paths for tmol components.
      
      Defined before other imports because JIT extension loading
      (tmol.utility.cpp_extension) imports this during module init.
      

