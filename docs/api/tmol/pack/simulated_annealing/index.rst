tmol.pack.simulated_annealing
=============================

.. py:module:: tmol.pack.simulated_annealing


Functions
---------

.. autoapisummary::

   tmol.pack.simulated_annealing.run_simulated_annealing


Module Contents
---------------

.. py:function:: run_simulated_annealing(energy_tables: tmol.pack.datatypes.PackerEnergyTables)

   .. rubric:: Docstring

   .. code-block:: text

      Run GPU simulated annealing.
      
      Phase 1 (hi-temp SA): 500 trajectories run at high temperature
      Phase 2 (lo-temp SA): Each top hi-temp trajectory seeds 10 lo-temp
      trajectories, then round1_cut = 0.25 keeps the top 25%
        -> 500 * 10 * 0.25 = 1250 trajectories
      Phase 3 (full quench): round2_cut = 0.25 keeps the top 25% of those
        -> int(1250 * 0.25) = 312 trajectories
      

