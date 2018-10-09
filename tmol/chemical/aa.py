import pandas
import toolz.functoolz


class AAIndex:
    """Indices for canonical, L amino acids

    This will eventually be where we store the mapping from L- to
    D-amino acids also (which biopython does not provide)"""

    _indices3 = {}
    _indices1 = {}

    @staticmethod
    @toolz.functoolz.memoize(cache=_indices3)
    def canonical_laa_ind3():
        """Convert upper-case 3-letter abbreviations of the 20 canonical L amino acids
        into an integer index in the range from 0 to 19."""
        import Bio.Data.IUPACData as bdiupacd

        return pandas.Index([x.upper() for x in bdiupacd.protein_letters_1to3.values()])

    @staticmethod
    @toolz.functoolz.memoize(cache=_indices1)
    def canonical_laa_ind1():
        """Convert upper-case 1-letter abbreviations of the 20 canonical L amino acids
        into an integer index in the range from 0 to 19, exactly matching the index
        assigned to the 3-letter abbreviations"""
        import Bio.Data.IUPACData as bdiupacd

        # ind3 = AAIndex.canonical_laa_ind3()
        return pandas.Index([x.upper() for x in bdiupacd.protein_letters_1to3.keys()])
