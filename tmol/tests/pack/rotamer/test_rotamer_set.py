import numpy
from tmol.pack.rotamer.rotamer_set import AASidechainBuilder
from tmol.system.io import ResidueReader


def test_aa_sidechain_builder_smoke():
    reader = ResidueReader.get_default()
    arg_restype = reader.residue_types["ARG"][0]
    arg_builder = AASidechainBuilder.from_restype(arg_restype)

    chi_for_ats = {
        "CG": 0,
        "1HB": 0,
        "2HB": 0,
        "CD": 1,
        "1HG": 1,
        "2HG": 1,
        "NE": 2,
        "1HD": 2,
        "2HD": 2,
        "CZ": 3,
        "HE": 3,
    }
    for i, at in enumerate(arg_restype.atoms):
        moving_chi = chi_for_ats.get(at.name, -1)
        assert moving_chi == arg_builder.chi_spins_dof[i]

    bbats = set(["N", "CA", "C", "O"])
    for i, at in enumerate(arg_restype.atoms):
        assert arg_builder.is_backbone_atom[i] == (at.name in bbats)
        if at.name in bbats:
            assert i in arg_builder.backbone_atom_inds

    # for resname, restype in reader.residue_types.items():
    #    pass
