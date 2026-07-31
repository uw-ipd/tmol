tmol.ligand.conformer_generation
================================

.. py:module:: tmol.ligand.conformer_generation

.. rubric:: Module docstring

.. code-block:: text

   Distance-geometry 3D coordinate generation for ligands from a SMILES.
   
   Replaces OpenBabel's ``make3D`` + rotor search. Pipeline: MMFF ideal geometry
   (RDKit, read-only) -> distance-bounds matrix -> metric-matrix embedding -> a
   torch stress refine (4D chiral annealing with random restarts, then a full-weight
   3D pass) -> OpenBabel force-field cleanup. MMFF-untypeable but modelable ligands
   (e.g. pentavalent phosphoranes) fall back to covalent-radius / hybridization
   ideals with explicit trigonal-bipyramidal constraints.
   
   The public entry :func:`generate_conformer` returns a coordinate-carrying pybel
   ``Molecule``; partial charges and atom names are assigned by the caller.
   


Attributes
----------

.. autoapisummary::

   tmol.ligand.conformer_generation.logger
   tmol.ligand.conformer_generation.W_EXACT
   tmol.ligand.conformer_generation.W_BOUND
   tmol.ligand.conformer_generation.W_PLANE
   tmol.ligand.conformer_generation.W_CHIRAL
   tmol.ligand.conformer_generation.CHIRAL_MARGIN
   tmol.ligand.conformer_generation.REFINE_ITERS
   tmol.ligand.conformer_generation.STAGE_A_ITERS
   tmol.ligand.conformer_generation.STAGE_A_ANNEAL
   tmol.ligand.conformer_generation.N_RESTART
   tmol.ligand.conformer_generation.TORCH_DTYPE


Functions
---------

.. autoapisummary::

   tmol.ligand.conformer_generation.generate_conformer


Module Contents
---------------

.. py:data:: logger

.. py:data:: W_EXACT
   :value: 50.0


.. py:data:: W_BOUND
   :value: 2.0


.. py:data:: W_PLANE
   :value: 10.0


.. py:data:: W_CHIRAL
   :value: 30.0


.. py:data:: CHIRAL_MARGIN
   :value: 4.0


.. py:data:: REFINE_ITERS
   :value: 200


.. py:data:: STAGE_A_ITERS
   :value: 15


.. py:data:: STAGE_A_ANNEAL
   :value: (0.0, 0.5, 5.0, 50.0)


.. py:data:: N_RESTART
   :value: 5


.. py:data:: TORCH_DTYPE
   :value: Ellipsis


.. py:function:: generate_conformer(smiles: str, *, minimize_steps: int = 50, seed: Optional[int] = None)

   .. rubric:: Docstring

   .. code-block:: text

      Conformer generation script SMILES -> pybel.
      
      Uses a custom generation scheme similar to RDKit's ETKDG, where an embedding
      + minimization in the embedding space is used to generate a very good guess
      at the initial conformer.  Uses pytorch minimization machinery, followed
      by a very short MMFF minimization.
      
      :param minimize_steps: OpenBabel conjugate-gradient steps for the final min.
      :param seed: Ooptional fixed RNG seed for reproducible coordinates.
      
      :raises ValueError: Failures in parsing or final min.
      

