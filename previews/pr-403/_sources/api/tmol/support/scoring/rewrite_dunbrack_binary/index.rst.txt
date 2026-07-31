tmol.support.scoring.rewrite_dunbrack_binary
============================================

.. py:module:: tmol.support.scoring.rewrite_dunbrack_binary


Attributes
----------

.. autoapisummary::

   tmol.support.scoring.rewrite_dunbrack_binary.rotamer_aliases
   tmol.support.scoring.rewrite_dunbrack_binary.parser


Functions
---------

.. autoapisummary::

   tmol.support.scoring.rewrite_dunbrack_binary.create_rotameric_data_for_aa
   tmol.support.scoring.rewrite_dunbrack_binary.strip_comments
   tmol.support.scoring.rewrite_dunbrack_binary.create_rotameric_aa_dunbrack_library
   tmol.support.scoring.rewrite_dunbrack_binary.create_semi_rotameric_aa_dunbrack_library
   tmol.support.scoring.rewrite_dunbrack_binary.create_dunbrack_rotamer_library


Module Contents
---------------

.. py:data:: rotamer_aliases

.. py:function:: create_rotameric_data_for_aa(aa_lines, nchi, rotamer_alias=None)

.. py:function:: strip_comments(lines)

.. py:function:: create_rotameric_aa_dunbrack_library(aa3, lines, nchi_for_aa, rotamer_alias)

.. py:function:: create_semi_rotameric_aa_dunbrack_library(aa3, nchi, bb_rotamer_lines, bbdep_density_lines, ref_bbdep_density_lines, bbind_rotamer_def_lines)

.. py:function:: create_dunbrack_rotamer_library(path_to_db_dir, path_to_reference_db_dir)

.. py:data:: parser

