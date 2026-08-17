

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    'TokenType': ('read_smiles', 'TokenType'),
    'LOGGER': ('smiles_helper', 'LOGGER'),
    'ISOTOPE_PATTERN': ('smiles_helper', 'ISOTOPE_PATTERN'),
    'ELEMENT_PATTERN': ('smiles_helper', 'ELEMENT_PATTERN'),
    'STEREO_PATTERN': ('smiles_helper', 'STEREO_PATTERN'),
    'HCOUNT_PATTERN': ('smiles_helper', 'HCOUNT_PATTERN'),
    'CHARGE_PATTERN': ('smiles_helper', 'CHARGE_PATTERN'),
    'CLASS_PATTERN': ('smiles_helper', 'CLASS_PATTERN'),
    'ATOM_PATTERN': ('smiles_helper', 'ATOM_PATTERN'),
    'VALENCES': ('smiles_helper', 'VALENCES'),
    'AROMATIC_ATOMS': ('smiles_helper', 'AROMATIC_ATOMS'),
    'parse_atom': ('smiles_helper', 'parse_atom'),
    'format_atom': ('smiles_helper', 'format_atom'),
    'parse_hcount': ('smiles_helper', 'parse_hcount'),
    'parse_charge': ('smiles_helper', 'parse_charge'),
    'bonds_missing': ('smiles_helper', 'bonds_missing'),
    'has_default_h_count': ('smiles_helper', 'has_default_h_count'),
    'mark_aromatic_atoms': ('smiles_helper', 'mark_aromatic_atoms'),
    'mark_aromatic_edges': ('smiles_helper', 'mark_aromatic_edges'),
    'increment_bond_orders': ('smiles_helper', 'increment_bond_orders'),
    'make_mol': ('testhelper', 'make_mol'),
    'assertEqualGraphs': ('testhelper', 'assertEqualGraphs'),
    'read_smiles': ('read_smiles', 'read_smiles'),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        import importlib
        mod_leaf, attr = _LAZY_ATTRS[name]
        mod = importlib.import_module(f".{mod_leaf}", package=__name__)
        # Re-cache every name from this module so that Python's import
        # machinery (which sets globals()[mod_leaf] = MODULE as a side-effect)
        # does not overwrite previously resolved function/class references.
        for _n, (_m, _a) in _LAZY_ATTRS.items():
            if _m == mod_leaf:
                try:
                    globals()[_n] = getattr(mod, _a)
                except AttributeError:
                    pass
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# -*- coding: utf-8 -*-
# Copyright 2018 Peter C Kroon

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PySMILES: The lightweight python module for reading and writing SMILES strings.
"""

from .read_smiles import read_smiles
from .write_smiles import write_smiles
from .smiles_helper import (fill_valence, add_explicit_hydrogens,
                            remove_explicit_hydrogens, correct_aromatic_rings)

