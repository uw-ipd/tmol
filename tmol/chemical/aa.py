import pandas
import Bio.Data.IUPACData


class AAIndex:
    """Indices for canonical, L amino acids"""

    canonical_laa_ind3 = pandas.Index(
        [x.upper() for x in Bio.Data.IUPACData.protein_letters_1to3.values()]
    )
