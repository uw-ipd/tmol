# -*- coding: utf-8 -*-
# Copyright 2018 Peter C Kroon
# Licensed under the Apache License, Version 2.0
"""PySMILES: The lightweight python module for reading and writing SMILES strings."""

from .read_smiles import read_smiles, TokenType  # noqa: F401
from .write_smiles import write_smiles  # noqa: F401
from .smiles_helper import (  # noqa: F401
    fill_valence,
    add_explicit_hydrogens,
    remove_explicit_hydrogens,
    correct_aromatic_rings,
    LOGGER,
    ISOTOPE_PATTERN,
    ELEMENT_PATTERN,
    STEREO_PATTERN,
    HCOUNT_PATTERN,
    CHARGE_PATTERN,
    CLASS_PATTERN,
    ATOM_PATTERN,
    VALENCES,
    AROMATIC_ATOMS,
    parse_atom,
    format_atom,
    parse_hcount,
    parse_charge,
    bonds_missing,
    has_default_h_count,
    mark_aromatic_atoms,
    mark_aromatic_edges,
    increment_bond_orders,
)
from .testhelper import make_mol, assertEqualGraphs  # noqa: F401
