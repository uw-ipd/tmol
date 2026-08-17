Packing API
===========

.. automodule:: tmol.pack.packer_task
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.pack_rotamers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.build_missing_sidechains
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.datatypes
   :members:
   :undoc-members:
   :show-inheritance:

Rotamer construction and samplers
---------------------------------

``PackerTask`` combines one or more conformer samplers. Protein workflows
normally use the Dunbrack sampler together with fixed-chi and current-conformer
sampling; nucleic-acid workflows add ``NaChiRotamerSampler``.

.. automodule:: tmol.pack.rotamer.build_rotamers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.rotamer.rotamer_set
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.rotamer.conformer_sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.rotamer.chi_sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.rotamer.dunbrack.dunbrack_chi_sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.rotamer.fixed_aa_chi_sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.rotamer.fallback_sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.rotamer.include_current_sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.rotamer.na_chi_sampler
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: tmol.pack.rotamer.opth_sampler
   :members:
   :undoc-members:
   :show-inheritance:
