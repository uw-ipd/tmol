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
# computing this distance exactly is slow) without affecting the score

MAX_SIG_BOND_SEPARATION = 6

# MAX_PATHS_FROM_CONNECTION:
#
# The maximum number of paths coming from a particular block type's connection.
# Currently this is 13 - 1 for the connection atom itself, 3 for the up-to-3 neighbors
# of that atom, and another 9 for the up-to-3 neighbors of each of those.
MAX_PATHS_FROM_CONNECTION = 13
