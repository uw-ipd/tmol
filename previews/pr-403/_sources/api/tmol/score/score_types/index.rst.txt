tmol.score.score_types
======================

.. py:module:: tmol.score.score_types


Classes
-------

.. autoapisummary::

   tmol.score.score_types.ScoreType


Module Contents
---------------

.. py:class:: ScoreType(*args, **kwds)

   Bases: :py:obj:`tmol.utility.auto_number.AutoNumber`


   .. rubric:: Docstring

   .. code-block:: text

      Create a collection of name/value pairs.
      
      Example enumeration:
      
      >>> class Color(Enum):
      ...     RED = 1
      ...     BLUE = 2
      ...     GREEN = 3
      
      Access them by:
      
      - attribute access:
      
        >>> Color.RED
        <Color.RED: 1>
      
      - value lookup:
      
        >>> Color(1)
        <Color.RED: 1>
      
      - name lookup:
      
        >>> Color['RED']
        <Color.RED: 1>
      
      Enumerations can be iterated over, and know how many members they have:
      
      >>> len(Color)
      3
      
      >>> list(Color)
      [<Color.RED: 1>, <Color.BLUE: 2>, <Color.GREEN: 3>]
      
      Methods can be added to enumerations, and members can have their own
      attributes -- see the documentation for details.
      

   .. py:attribute:: fa_ljatr
      :value: ()



   .. py:attribute:: fa_ljrep
      :value: ()



   .. py:attribute:: fa_lk
      :value: ()



   .. py:attribute:: fa_elec
      :value: ()



   .. py:attribute:: hbond
      :value: ()



   .. py:attribute:: cart_lengths
      :value: ()



   .. py:attribute:: cart_angles
      :value: ()



   .. py:attribute:: cart_torsions
      :value: ()



   .. py:attribute:: cart_impropers
      :value: ()



   .. py:attribute:: cart_hxltorsions
      :value: ()



   .. py:attribute:: constraint
      :value: ()



   .. py:attribute:: disulfide
      :value: ()



   .. py:attribute:: omega
      :value: ()



   .. py:attribute:: rama
      :value: ()



   .. py:attribute:: dunbrack_rot
      :value: ()



   .. py:attribute:: dunbrack_rotdev
      :value: ()



   .. py:attribute:: dunbrack_semirot
      :value: ()



   .. py:attribute:: lk_ball_iso
      :value: ()



   .. py:attribute:: lk_ball
      :value: ()



   .. py:attribute:: lk_bridge
      :value: ()



   .. py:attribute:: lk_bridge_uncpl
      :value: ()



   .. py:attribute:: ref
      :value: ()



   .. py:attribute:: gen_torsions
      :value: ()



   .. py:attribute:: n_score_types
      :value: ()



