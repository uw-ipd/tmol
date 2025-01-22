# import attr
# import cattr

# import numpy
# import pandas

# from typing import Sequence, Tuple

# from tmol.types.array import NDArray

# from tmol.chemical.restypes import Residue
# from .datatypes import (
#     atom_metadata_dtype,
#     torsion_metadata_dtype,
#     connection_metadata_dtype,
# )


# def _ceil_to_size(size, val):
#     d, m = numpy.divmod(val, size)
#     return (d + (m != 0).astype(int)) * size


# @attr.s(auto_attribs=True, slots=True)
# class PackedResidueSystem:
#     # coord buffer block size, residue start indices are aligned to block boundaries
#     block_size: int

#     # residue objects packed into system
#     residues: Sequence[Residue]

#     # residue start indicies within `coords`
#     res_start_ind: NDArray[int][:]

#     # total system size
#     system_size: int

#     # atomic coordinate buffer, nan-filled in 'unused' regions
#     coords: NDArray[float][:, 3]

#     # inter-atomic bond indices
#     bonds: NDArray[int][:, 2]

#     atom_metadata: NDArray[atom_metadata_dtype][:]
#     torsion_metadata: NDArray[torsion_metadata_dtype][:]
#     connection_metadata: NDArray[connection_metadata_dtype][:]

#     @classmethod
#     def from_residues(cls, res: Sequence[Residue], block_size=8):
#         """Initialize a packed residue system from list of residue containers."""

#         ### Pack residues within the coordinate system

#         # Align residue starts to block boundries
#         #
#         # Ceil each residue's size to generate residue segments
#         res_lengths = numpy.array([len(r.coords) for r in res])
#         res_segment_lengths = _ceil_to_size(block_size, res_lengths)

#         # Ensure that the -1 index is a valid "null atom"
#         # padding with a null block if needed.
#         if (res_lengths[-1] % block_size) == 0:
#             res_segment_lengths[-1] += block_size

#         segment_ends = res_segment_lengths.cumsum()
#         segment_starts = numpy.empty_like(segment_ends)
#         segment_starts[0] = 0
#         segment_starts[1:] = segment_ends[:-1]

#         res_aidx = segment_starts

#         # Allocate the coordinate system and attach residues
#         buffer_size = segment_ends[-1]
#         cbuff = numpy.full((buffer_size, 3), numpy.nan)

#         attached_res = [
#             r.attach_to(cbuff[start : start + len(r.coords)])
#             for r, start in zip(res, res_aidx)
#         ]

#         ### Generate atom metadata

#         atom_metadata = numpy.empty(buffer_size, atom_metadata_dtype)
#         atom_metadata["atom_index"] = numpy.arange(len(atom_metadata))
#         atom_metadata["residue_index"] = None

#         for ri, (rs, r) in enumerate(zip(res_aidx, res)):
#             rt = r.residue_type
#             residue_block = atom_metadata[rs : rs + len(rt.atoms)]
#             residue_block["residue_name"] = rt.name
#             residue_block["atom_name"] = [a.name for a in rt.atoms]
#             residue_block["atom_type"] = [a.atom_type for a in rt.atoms]
#             # TODO Rewrite chemical database with explict element tag.
#             residue_block["atom_element"] = [a.atom_type[0] for a in rt.atoms]
#             residue_block["residue_index"] = ri

#         ### Index residue connectivity
#         # Generate a table of residue connections, with "from" and "to" entries
#         # for *both* directions across the connection.
#         #
#         # Just a linear set of connections up<->down for now.
#         residue_connections = []

#         for i, j in zip(range(len(res) - 1), range(1, len(res))):
#             valid_connection = (
#                 "up" in res[i].residue_type.connection_to_idx
#                 and "down" in res[j].residue_type.connection_to_idx
#             )

#             if valid_connection:
#                 residue_connections.extend(
#                     [(i, "up", i + 1, "down"), (i + 1, "down", i, "up")]
#                 )
#             else:
#                 # TODO add logging
#                 pass

#         for f_i, f_n, t_i, t_n in residue_connections:
#             assert (
#                 f_n in res[f_i].residue_type.connection_to_idx
#             ), f"residue missing named connection: {f_n!r} res:\n{res[f_i]}"
#             assert (
#                 t_n in res[t_i].residue_type.connection_to_idx
#             ), f"residue missing named connection: {t_n!r} res:\n{res[t_i]}"

#         connection_index = pandas.DataFrame.from_records(
#             residue_connections,
#             columns=pandas.MultiIndex.from_tuples(
#                 [("from", "resi"), ("from", "cname"), ("to", "resi"), ("to", "cname")]
#             ),
#         )

#         # Unpack the connection metadata table
#         connection_metadata = numpy.empty(
#             len(connection_index), dtype=connection_metadata_dtype
#         )

#         connection_metadata["from_residue_index"] = connection_index["from"]["resi"]
#         connection_metadata["from_connection_name"] = connection_index["from"]["cname"]

#         connection_metadata["to_residue_index"] = connection_index["to"]["resi"]
#         connection_metadata["to_connection_name"] = connection_index["to"]["cname"]

#         # Generate an index of all the connection atoms in the system,
#         # resolving the internal and global index of the connection atoms
#         connection_atoms = pandas.DataFrame.from_records(
#             [
#                 (ri, cname, c_aidx, c_aidx + r_g_aidx)
#                 for (ri, (r_g_aidx, r)) in enumerate(zip(res_aidx, res))
#                 for cname, c_aidx in r.residue_type.connection_to_idx.items()
#             ],
#             columns=["resi", "cname", "internal_aidx", "aidx"],
#         )

#         # Merge against the connection table to generate a connection entry
#         # with the residue index, the connection name, the local atom index,
#         # and the global atom index for the connection by merging on the
#         # "cname", "resi" columns.
#         #
#         # columns:
#         # cname resi internal_aidx  aidx
#         from_connections = pandas.merge(connection_index["from"], connection_atoms)
#         to_connections = pandas.merge(connection_index["to"], connection_atoms)

#         for c in from_connections.columns:
#             connection_index["from", c] = from_connections[c]
#         for c in to_connections.columns:
#             connection_index["to", c] = to_connections[c]

#         ### Generate the bond graph

#         # Offset the internal bond graph by the residue start idx
#         intra_res_bonds = numpy.concatenate(
#             [
#                 r.residue_type.bond_indices + start
#                 for start, r in zip(segment_starts, res)
#             ]
#         )

#         # Join the connection global atom indices
#         inter_res_bonds = numpy.vstack(
#             [from_connections["aidx"].values, to_connections["aidx"].values]
#         ).T

#         bonds = numpy.concatenate([intra_res_bonds, inter_res_bonds])

#         ### Generate dihedral metadata for all named torsions

#         # Generate a lookup from residue/connection name to connected residue
#         connection_lookup = pandas.concat(
#             (
#                 pandas.DataFrame(  # All the named connections
#                     dict(
#                         residue_index=connection_index["from", "resi"],
#                         from_cname=connection_index["from", "cname"],
#                         to_residue=connection_index["to", "resi"],
#                         to_cname=connection_index["to", "cname"],
#                     )
#                 ),
#                 pandas.DataFrame(  # Loop-back to self for "None" connections
#                     dict(
#                         from_cname=None,
#                         residue_index=numpy.arange(len(res)),
#                         to_residue=numpy.arange(len(res)),
#                         to_cname=None,
#                     )
#                 ),
#             ),
#             ignore_index=True,
#             sort=True,
#         )

#         downstream_atoms = []
#         for i, r in enumerate(res):
#             n_atoms = r.coords.shape[0]
#             n_conns = len(r.residue_type.connections)
#             inds0 = numpy.arange(n_conns * n_atoms, dtype=numpy.int32) // n_atoms
#             inds1 = numpy.arange(n_conns * n_atoms, dtype=numpy.int32) % n_atoms

#             downstream_atoms.append(
#                 pandas.DataFrame(
#                     dict(
#                         residue_index=numpy.full(
#                             (n_atoms * n_conns,), i, dtype=numpy.int32
#                         ),
#                         conn=numpy.array([c.name for c in r.residue_type.connections])[
#                             inds0
#                         ],
#                         downstream_distance=inds1,
#                         downstream_atom_index=(
#                             segment_starts[i]
#                             + r.residue_type.atom_downstream_of_conn[inds0, inds1]
#                         ),
#                     )
#                 )
#             )
#         downstream_atoms = pandas.concat(downstream_atoms)

#         # Generate a lookup from residue index and atom name to global atom index.
#         atom_lookup = pandas.DataFrame(
#             dict(
#                 residue_index=atom_metadata["residue_index"],
#                 atom_name=atom_metadata["atom_name"],
#                 atom_index=numpy.arange(len(atom_metadata)),
#             )
#         )
#         atom_lookup = atom_lookup[~atom_lookup.residue_index.isna()]

#         # Unpack all the residue type torsion entries, and tag with the
#         # source residue index
#         torsion_entries = [
#             dict(residue_index=ri, **torsion_entry)
#             for ri, r in enumerate(res)
#             for torsion_entry in cattr.unstructure(r.residue_type.torsions)
#         ]

#         if torsion_entries:
#             torsion_entries = pandas.json_normalize(torsion_entries)
#             for column_letter in ("a", "b", "c", "d"):
#                 colname = "%s.bond_sep_from_conn" % column_letter
#                 col = numpy.array(torsion_entries[colname])
#                 col[numpy.logical_not(numpy.any(col, axis=0))] = -1
#                 col = col.astype(float)
#                 col[numpy.isnan(col)] = -1
#                 col = col.astype(numpy.int32)
#                 torsion_entries[colname] = col

#             # split the torsion_entries data frame up into the four individual torsions
#             # then divide these based on whether they span to another residue, or
#             # whether they are intra-residue.
#             #
#             # merge the inter-residue entries with the
#             torsion_entries_by_atom = []
#             for atom in ("a", "b", "c", "d"):
#                 col_names = [
#                     "residue_index",
#                     "name",
#                     "%s.atom" % atom,
#                     "%s.connection" % atom,
#                     "%s.bond_sep_from_conn" % atom,
#                 ]
#                 sub_df = torsion_entries[col_names]
#                 sub_df_inter = sub_df[pandas.notna(sub_df["%s.connection" % atom])]
#                 sub_df_intra = sub_df[pandas.notna(sub_df["%s.atom" % atom])]

#                 # now lets do a left merge on the inter-residue rows
#                 # against the connection-lookup dataframe to determine the
#                 # residue that is connected and the name of the connection point
#                 # on that residue to which the connection on the source residue
#                 # is connected. Next merge against the downstream_atoms dataframe
#                 # to determine the number of bonds separating the

#                 sub_df_inter = pandas.merge(
#                     sub_df_inter,
#                     connection_lookup.rename(
#                         columns={
#                             "from_cname": "%s.connection" % atom,
#                             "to_residue": "%s.residue" % atom,
#                             "to_cname": "%s.to_connection" % atom,
#                         }
#                     ),
#                     how="left",
#                     copy=False,
#                 )
#                 sub_df_inter = pandas.merge(
#                     sub_df_inter,
#                     downstream_atoms.rename(
#                         columns={
#                             "residue_index": "%s.residue" % atom,
#                             "conn": "%s.to_connection" % atom,
#                             "downstream_distance": "%s.bond_sep_from_conn" % atom,
#                             "downstream_atom_index": "%s.atom_index" % atom,
#                         }
#                     ),
#                     how="left",
#                     copy=False,
#                 )[["residue_index", "name", "%s.atom_index" % atom]]

#                 # now lets do a left merge on the intra-residue rows against
#                 # the atom-lookup dataframe to go directly to the atom index
#                 # for atoms within a residue
#                 sub_df_intra = pandas.merge(
#                     sub_df_intra,
#                     atom_lookup.rename(
#                         columns={
#                             "atom_name": "%s.atom" % atom,
#                             "atom_index": "%s.atom_index" % atom,
#                         }
#                     ),
#                     how="left",
#                     copy=False,
#                 )[["residue_index", "name", "%s.atom_index" % atom]]

#                 concatted = pandas.concat([sub_df_inter, sub_df_intra])
#                 torsion_entries_by_atom.append(concatted)

#             merge = torsion_entries_by_atom[0]
#             for i in range(3):
#                 merge = pandas.merge(
#                     merge, torsion_entries_by_atom[i + 1], how="left", copy=False
#                 )
#             torsion_index = merge.sort_index(axis="columns")

#         else:
#             torsion_index = pandas.DataFrame(
#                 {
#                     "residue_index": numpy.empty(0, float),
#                     "name": numpy.empty(0, object),
#                     "a.atom_index": numpy.empty(0, float),
#                     "b.atom_index": numpy.empty(0, float),
#                     "c.atom_index": numpy.empty(0, float),
#                     "d.atom_index": numpy.empty(0, float),
#                 }
#             )

#         # print("torsion_index")
#         # print(torsion_index)
#         pandas.DataFrame

#         # Unpack the merge frame into atomic indices, fixing up any missing values to
#         # point to the "-1" atom, which will have nan coordinates.
#         def nan_to_neg1(v):
#             return numpy.where(~numpy.isnan(v), v, -1).astype(int)

#         torsion_metadata = numpy.empty(len(torsion_index), torsion_metadata_dtype)
#         torsion_metadata["residue_index"] = torsion_index["residue_index"]
#         torsion_metadata["name"] = torsion_index["name"]
#         torsion_metadata["atom_index_a"] = nan_to_neg1(torsion_index["a.atom_index"])
#         torsion_metadata["atom_index_b"] = nan_to_neg1(torsion_index["b.atom_index"])
#         torsion_metadata["atom_index_c"] = nan_to_neg1(torsion_index["c.atom_index"])
#         torsion_metadata["atom_index_d"] = nan_to_neg1(torsion_index["d.atom_index"])

#         result = cls(
#             block_size=int(block_size),
#             system_size=int(buffer_size),
#             res_start_ind=segment_starts,
#             residues=attached_res,
#             atom_metadata=atom_metadata,
#             torsion_metadata=torsion_metadata,
#             connection_metadata=connection_metadata,
#             bonds=bonds,
#             coords=cbuff,
#         )

#         return result


# @attr.s(auto_attribs=True, slots=True)
# class PackedResidueSystemStack:
#     systems: Tuple[PackedResidueSystem, ...]
