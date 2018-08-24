from collections import Counter


def test_ljlk_defs(default_database):

    db = default_database.scoring.ljlk
    atom_type_counts = Counter(n.name for n in db.atom_type_parameters)

    for at in atom_type_counts:
        assert atom_type_counts[at] == 1, f"Duplicate ljlk type parameter: {at}"

    for at in db.atom_type_parameters:
        assert at.name in default_database.chemical.atom_types
