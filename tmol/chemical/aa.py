import pandas
import toolz.functoolz


class AAIndex:
    """Indices for canonical, L amino acids"""

    _indices = {}

    @staticmethod
    @toolz.functoolz.memoize(cache=_indices)
    def canonical_laa_ind3():
        """Convert upper-case 3-letter abbreviations of the 20 canonical L amino acids
        into an integer index in the range from 0 to 19."""
        import Bio.Data.IUPACData as bdiupacd

        return pandas.Index([x.upper() for x in bdiupacd.protein_letters_1to3.values()])
