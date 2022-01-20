import numpy

partial_atom_id_dtype = numpy.dtype(
    [("atom_id", int), ("resconn_id", int), ("bonds_from_resconn", int)]
)


atom_metadata_dtype = numpy.dtype(
    [
        ("residue_name", object),
        ("atom_name", object),
        ("atom_type", object),
        ("atom_element", object),
        ("atom_index", object),
        ("residue_index", float),
    ]
)

torsion_metadata_dtype = numpy.dtype(
    [
        ("residue_index", int),
        ("name", object),
        ("atom_index_a", int),
        ("atom_index_b", int),
        ("atom_index_c", int),
        ("atom_index_d", int),
    ]
)

connection_metadata_dtype = numpy.dtype(
    [
        ("from_residue_index", int),
        ("from_connection_name", object),
        ("to_residue_index", int),
        ("to_connection_name", object),
    ]
)
