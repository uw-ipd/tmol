tmol.ligand.mol3d
=================

.. py:module:: tmol.ligand.mol3d

.. rubric:: Module docstring

.. code-block:: text

   Authoritative partial-charge mapping for prepared ligands.
   
   The unified ligand path derives every ligand's partial charges from its SMILES
   via OpenBabel MMFF94 (see
   :func:`tmol.ligand.detect.nonstandard_residue_info_from_smiles_via_mol2`). Those
   charges arrive on the detected ligand as an ``{atom_name: charge}`` map in
   source-atom order. This module maps them onto the prepared molecule purely by
   stable atom index, so charges are wholly independent of any later atom renaming
   and no force-field recomputation is ever attempted.
   
   If authoritative charges are missing, incomplete, or mis-sized, preparation
   fails loudly rather than guessing -- there is no RDKit/Gasteiger fallback.
   


Functions
---------

.. autoapisummary::

   tmol.ligand.mol3d.authoritative_charges_by_index


Module Contents
---------------

.. py:function:: authoritative_charges_by_index(source_atom_names: Sequence[str], partial_charges: Optional[Mapping[str, float]], mol: rdkit.Chem.Mol, *, ligand_name: str = '') -> dict[int, float]

   .. rubric:: Docstring

   .. code-block:: text

      Return ``{atom_index: charge}`` mapping source charges onto ``mol``.
      
      ``source_atom_names[i]`` must name atom ``i`` of ``mol``. The SMILES -> mol2
      reader preserves atom order from the OpenBabel mol2 through to the prepared
      molecule, so the per-atom MMFF94 charges can be applied directly by index --
      independent of any downstream atom renaming.
      
      :param source_atom_names: Atom names in source (OpenBabel mol2) order.
      :param partial_charges: Authoritative ``{atom_name: charge}`` map from the
                              SMILES -> OpenBabel MMFF94 step.
      :param mol: The prepared RDKit molecule (same atom order as the source).
      :param ligand_name: Optional residue name for error messages.
      
      :returns: ``{rdkit_atom_index: partial_charge}`` for every atom in ``mol``.
      
      :raises ValueError: If charges are absent, incomplete, or atom counts disagree.
      

