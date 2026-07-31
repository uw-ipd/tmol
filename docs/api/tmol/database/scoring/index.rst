tmol.database.scoring
=====================

.. py:module:: tmol.database.scoring


Submodules
----------

.. toctree::
   :maxdepth: 1

   /api/tmol/database/scoring/cartbonded/index
   /api/tmol/database/scoring/disulfide/index
   /api/tmol/database/scoring/dunbrack_libraries/index
   /api/tmol/database/scoring/elec/index
   /api/tmol/database/scoring/genbonded/index
   /api/tmol/database/scoring/hbond/index
   /api/tmol/database/scoring/ljlk/index
   /api/tmol/database/scoring/omega_bbdep/index
   /api/tmol/database/scoring/rama/index
   /api/tmol/database/scoring/ref/index


Classes
-------

.. autoapisummary::

   tmol.database.scoring.ScoringDatabase


Package Contents
----------------

.. py:class:: ScoringDatabase

   .. py:attribute:: cartbonded
      :type:  cartbonded.CartBondedDatabase


   .. py:attribute:: genbonded
      :type:  genbonded.GenBondedDatabase


   .. py:attribute:: disulfide
      :type:  disulfide.DisulfideDatabase


   .. py:attribute:: dun
      :type:  dunbrack_libraries.DunbrackRotamerLibrary


   .. py:attribute:: elec
      :type:  elec.ElecDatabase


   .. py:attribute:: hbond
      :type:  hbond.HBondDatabase


   .. py:attribute:: ljlk
      :type:  ljlk.LJLKDatabase


   .. py:attribute:: omega_bbdep
      :type:  omega_bbdep.OmegaBBDepDatabase


   .. py:attribute:: rama
      :type:  rama.RamaDatabase


   .. py:attribute:: ref
      :type:  ref.RefDatabase


   .. py:method:: from_file(path=os.path.dirname(__file__))
      :classmethod:



