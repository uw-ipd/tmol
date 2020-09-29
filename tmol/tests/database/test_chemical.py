import tmol.database


def test_residue_defs(default_database: tmol.database.ParameterDatabase):

    atom_types = set(at.name for at in default_database.chemical.atom_types)
    assert len(default_database.chemical.atom_types) == len(
        atom_types
    ), "Duplicate atom types."

    for r in default_database.chemical.residues:
        for a in r.atoms:
            assert (
                a.atom_type in atom_types
            ), f"Invalid atom type. res: {r.name3} atom: {a}"

        atom_names = {a.name for a in r.atoms}
        assert len(atom_names) == len(r.atoms), "atom names not unique."

        connection_names = {c.name for c in r.connections}
        assert len(connection_names) == len(
            r.connections
        ), "connection names not unique."

        # Check that atom and icoor names are unique
        valid_icoor_names = set.union(atom_names, connection_names)
        assert len(valid_icoor_names) == len(r.atoms) + len(
            r.connections
        ), "atom/icoor name collision"

        # Check that all icoors reference a valid target
        for icoor in r.icoors:
            assert icoor.name in valid_icoor_names
            assert icoor.parent in valid_icoor_names
            assert icoor.grand_parent in valid_icoor_names
            assert icoor.great_grand_parent in valid_icoor_names

        torsion_names = {t.name for t in r.torsions}
        assert len(torsion_names) == len(r.torsions), "torsion names not unique"

        for t in r.torsions:
            for catom in (t.a, t.b, t.c, t.d):
                assert (
                    catom.atom in atom_names
                    and catom.connection is None
                    and catom.bond_sep_from_conn is None
                ) or (
                    catom.atom is None
                    and catom.connection in connection_names
                    and catom.bond_sep_from_conn is not None
                )
        if r.properties.polymer.mainchain_atoms is not None:
            for at in r.properties.polymer.mainchain_atoms:
                assert at in atom_names
