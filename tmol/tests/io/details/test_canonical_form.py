from tmol.io.canonical_ordering import (
    max_n_canonical_atoms,
    canonical_form_from_pdb_lines,
)


def test_canonical_form_from_pdb_lines(pertuzumab_lines):
    chain_begin, res_types, coords, atom_is_present = canonical_form_from_pdb_lines(
        pertuzumab_lines
    )
    assert chain_begin.shape[0] == res_types.shape[0]
    assert chain_begin.shape[0] == coords.shape[0]
    assert chain_begin.shape[0] == atom_is_present.shape[0]
    assert chain_begin.shape[1] == res_types.shape[1]
    assert chain_begin.shape[1] == coords.shape[1]
    assert chain_begin.shape[1] == atom_is_present.shape[1]
    assert atom_is_present.shape[2] == max_n_canonical_atoms
    assert coords.shape[2] == max_n_canonical_atoms
    assert coords.shape[3] == 3
