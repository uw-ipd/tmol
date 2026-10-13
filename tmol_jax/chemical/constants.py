"""Chemical constants for tmol_jax.

Ported verbatim from ``tmol.chemical.constants``.
"""

from __future__ import annotations

from typing import Dict

# MAX_SIG_BOND_SEPARATION: The maximum significant bond separation.
#
# The maximum number of chemical bonds separating any pair of atoms that we would
# need to know s.t. if the number of chemical bonds between that pair is less
# than this value, we would need to know that number exactly, and if the number
# is greater than or equal to this value, then, for our purposes, knowing that
# number exactly is not more informative than knowing it is not less than this
# maximum.
#
# e.g. The C atom on residue i and the N atom on residue i+2 are 4 chemical bonds
# apart. Knowing this exactly impacts how these two atoms are scored against each
# other. The N atom on residue i and the C atom on residue i+2 are 8 chemical bonds
# apart; no part of scoring will differentiate between an atom pair 8 chemical bonds
# apart and a pair that is 6 chemical bonds apart, so if MAX_SACS is 6, then we
# can say the separation between this second pair of atoms is 6 (perhaps because
# computing this distance exactly is slow) without affecting the score.
MAX_SIG_BOND_SEPARATION: int = 6


# MAX_PATHS_FROM_CONNECTION:
#
# The maximum number of paths coming from a particular block type's connection.
# Currently this is 13: 1 for the connection atom itself, 3 for the up-to-3 neighbors
# of that atom, and another 9 for the up-to-3 neighbors of each of those.
#
# Note: in the future, if we change the maximum number of possible bonds to other atoms,
# this will need to change. Per aleaverfay, the summation would be
# sum( 0 <= i <= 2, (MAX_N_BONDS_TO_OTHER_ATOMS - 1) ^ i )
# where MAX_N_BONDS_TO_OTHER_ATOMS is 4.
MAX_PATHS_FROM_CONNECTION: int = 13


# Canonical three-letter codes for the 20 proteinogenic amino acids
# and their one-letter shorthand counterparts. These mappings are used by
# upstream downstream io and scoring code and are included here as top-level
# attributes so callers can ``from tmol_jax.chemical.constants import one2three``.
one2three: Dict[str, str] = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


three2one: Dict[str, str] = {v: k for k, v in one2three.items()}


#: Canonical amino acid three-letter list (in the insertion order of ``one2three``).
canonical_aa_3letter: tuple = tuple(one2three.values())


#: Canonical amino acid one-letter list.
canonical_aa_1letter: tuple = tuple(one2three.keys())


__all__ = [
    "MAX_SIG_BOND_SEPARATION",
    "MAX_PATHS_FROM_CONNECTION",
    "one2three",
    "three2one",
    "canonical_aa_3letter",
    "canonical_aa_1letter",
]
