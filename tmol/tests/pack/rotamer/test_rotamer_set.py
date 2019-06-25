import numpy
from tmol.pack.rotamer.rotamer_set import SingleSidechainBuilder
from tmol.system.io import ResidueReader
from tmol.database.chemical import SidechainBuilding
import attr


def test_sidechain_builder_determine_sidechain_group(default_database):
    reader = ResidueReader.get_default()
    arg_restype = reader.residue_types["ARG"][0]
    keep, nonsc, dfs_inds = SingleSidechainBuilder.determine_atoms_in_sidechain_group(
        default_database.chemical, arg_restype, 0
    )


def test_aa_sidechain_builder(default_database):
    reader = ResidueReader.get_default()
    arg_restype = reader.residue_types["ARG"][0]
    arg_builder = SingleSidechainBuilder.from_restype(
        default_database.chemical, arg_restype, 0
    )

    assert arg_builder.natoms == len(arg_restype.atoms)

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
        assert moving_chi == arg_builder.chi_that_spins_atom[i]

    bbats = set(["N", "CA", "C", "O"])
    for i, at in enumerate(arg_restype.atoms):
        assert arg_builder.is_backbone_atom[i] == (at.name in bbats)
        if at.name in bbats:
            assert i in arg_builder.backbone_atom_inds

    assert arg_builder.sidechain_root.shape[0] == 3
    assert arg_builder.sidechain_root[0] == arg_restype.atom_to_idx["CB"]
    assert arg_builder.sidechain_root[1] == arg_restype.atom_to_idx["CA"]
    assert arg_builder.sidechain_root[2] == arg_restype.atom_to_idx["N"]

    # for resname, restype in reader.residue_types.items():
    #    pass


def test_aa_sidechain_builder_w_missing_ats(default_database):
    reader = ResidueReader.get_default()
    arg_restype = reader.residue_types["ARG"][0]
    arg_restype2 = attr.evolve(
        arg_restype,
        sidechain_building=[SidechainBuilding(root="CB", backbone_atoms=["N", "CA"])],
    )

    arg_builder = SingleSidechainBuilder.from_restype(
        default_database.chemical, arg_restype, 0
    )
    arg_builder2 = SingleSidechainBuilder.from_restype(
        default_database.chemical, arg_restype2, 0
    )

    assert arg_builder2.natoms == len(arg_restype.atoms) - 2

    assert len(arg_builder2.backbone_atom_inds) == 2
    for i in range(arg_builder2.natoms):
        ires = arg_builder2.rotatom_2_resatom[i].item()
        assert arg_builder.is_backbone_atom[ires] == arg_builder2.is_backbone_atom[i]
        for j in range(arg_builder2.natoms):
            jres = arg_builder2.rotatom_2_resatom[j]
            assert arg_builder.bonds[ires, jres] == arg_builder2.bonds[i, j]

        numpy.testing.assert_allclose(
            arg_builder.atom_icoors[ires, :].numpy(),
            arg_builder2.atom_icoors[i, :].numpy(),
        )

        # print(i, ires, arg_builder.chi_that_spins_atom[ires], arg_builder2.chi_that_spins_atom[i])
        assert (
            arg_builder.chi_that_spins_atom[ires] == arg_builder2.chi_that_spins_atom[i]
        )

    excluded = ["C", "O"]
    for at in excluded:
        atind = arg_restype.atom_to_idx[at]
        assert arg_builder2.resatom_2_rotatom[atind] == -1

    for i in arg_builder2.sidechain_dfs:
        assert 0 <= i and i < arg_builder2.natoms

    # for resname, restype in reader.residue_types.items():
    #    pass
