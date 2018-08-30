import tmol.chemical.aa as aa
import Bio.Data.IUPACData as biopy_iupac


def test_all_lcaa_res3_in_index():
    ind3 = aa.AAIndex.canonical_laa_ind3()
    # 20 canonical amino acids
    assert len(ind3) == 20

    for aa3 in biopy_iupac.protein_letters_1to3.values():
        assert aa3.upper() in ind3


def test_all_lcaa_res1_in_index():
    ind1 = aa.AAIndex.canonical_laa_ind1()
    # 20 canonical amino acids
    assert len(ind1) == 20

    for aa1 in biopy_iupac.protein_letters_1to3.keys():
        assert aa1.upper() in ind1


def test_all_lcaa_res1_ind_matches_res3_ind():
    ind3 = aa.AAIndex.canonical_laa_ind3()
    ind1 = aa.AAIndex.canonical_laa_ind1()
    for aa1, aa3 in biopy_iupac.protein_letters_1to3.items():
        assert ind1.get_loc(aa1.upper()) == ind3.get_loc(aa3.upper())
