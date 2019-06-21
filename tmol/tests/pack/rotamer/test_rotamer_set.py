import numpy
from tmol.score.pack.rotamer_set import AASidechainBuilder
from tmol.system.io import ResidueReader


def test_aa_sidechain_builder_smoke():
    reader = ResidueReader.get_default()
    arg_restype = reader.residue_types["ARG"]
    arg_builder = AASidechainBuilder(arg_restype)

    # for resname, restype in reader.residue_types.items():
    #    pass
