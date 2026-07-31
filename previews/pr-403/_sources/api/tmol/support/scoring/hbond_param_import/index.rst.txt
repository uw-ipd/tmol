tmol.support.scoring.hbond_param_import
=======================================

.. py:module:: tmol.support.scoring.hbond_param_import

.. rubric:: Module docstring

.. code-block:: text

   Parse and import rosetta hydrogen bond parameters.
   
   Manages parsing and import a subset of rosetta hydrogen bond parameters into a
   hydrogen bond parameter database file. Selects a minimal subset of polynomial
   parameters and type pair parameters to cover a specificed set of donor/acceptor
   types.
   
   Example::
   
       with open("sp2_elec_hbond_params.yaml", "w") as outfile:
           params = RosettaHBParams(
               "~/workspace/rosetta/main/"
               "database/scoring/score_functions/hbonds/sp2_elec_params/"
           )
           params.to_yaml(outfile)
   


Attributes
----------

.. autoapisummary::

   tmol.support.scoring.hbond_param_import.table_schema
   tmol.support.scoring.hbond_param_import.RawParams
   tmol.support.scoring.hbond_param_import.basetype_for_dtype


Classes
-------

.. autoapisummary::

   tmol.support.scoring.hbond_param_import.RosettaHBParams


Functions
---------

.. autoapisummary::

   tmol.support.scoring.hbond_param_import.attrs_for_dtypes


Module Contents
---------------

.. py:data:: table_schema

.. py:data:: RawParams

.. py:class:: RosettaHBParams

   .. py:attribute:: target_donors
      :value: ('hbdon_PBA', 'hbdon_CXA', 'hbdon_IMD', 'hbdon_IME', 'hbdon_IND', 'hbdon_AMO', 'hbdon_GDE',...



   .. py:attribute:: target_acceptors
      :value: ('hbacc_PBA', 'hbacc_CXA', 'hbacc_CXL', 'hbacc_IMD', 'hbacc_IME', 'hbacc_AHX', 'hbacc_HXL', 'hbacc_H2O')



   .. py:attribute:: path
      :type:  str


   .. py:attribute:: tables
      :type:  RawParams


   .. py:attribute:: donor_types
      :type:  pandas.DataFrame


   .. py:attribute:: acceptor_types
      :type:  pandas.DataFrame


   .. py:attribute:: pair_params
      :type:  pandas.DataFrame


   .. py:attribute:: polynomial_parameters
      :type:  pandas.DataFrame


   .. py:method:: to_yaml(outfile=None)


.. py:data:: basetype_for_dtype

.. py:function:: attrs_for_dtypes(name, dtypes)

