import toolz
import cattr

from properties import HasProperties, List, Integer, Instance
from tmol.properties.array import Array

import numpy
import pandas

from typing import Sequence

from .restypes import Residue


class PackedResidueSystem(HasProperties):

    block_size: int = Integer(
        "coord buffer block size, residue start indicies are aligned to block boundries",
        default=8,
        min=1,
        cast=True
    )

    residues: Sequence[Residue] = List(
        "residue objects packed into system",
        prop=Instance("attached residue", Residue)
    )

    res_start_ind: Sequence[int] = Array(
        "residue start indicies within `coords`", dtype=int
    )[:]

    system_size: int = Integer("total system size")

    coords: numpy.ndarray = Array(
        "atomic coordinate buffer, nan-filled in 'unused' regions",
        dtype=float,
        cast="unsafe"
    )[:, 3]

    bonds: numpy.ndarray = Array(
        "inter-atomic bond indices", dtype=int, cast="unsafe"
    )[:, 2]

    atom_metadata_dtype = numpy.dtype([
        ("residue_name", object),
        ("atom_name", object),
        ("atom_type", object),
        ("atom_index", object),
        ("residue_index", float),
    ])

    atom_metadata: numpy.ndarray = Array(
        "atom metada", dtype=atom_metadata_dtype
    )[:]

    @staticmethod
    def _ceil_to_size(size, val):
        d, m = numpy.divmod(val, size)
        return (d + (m != 0).astype(int)) * size

    def from_residues(self, res: Sequence[Residue]):
        """Initialize a packed residue system from list of residue containers."""

        ### Pack residues within the coordinate system

        # Align residue starts to block boundries
        #
        # Ceil each residue's size to generate residue segments
        res_lengths = numpy.array([len(r.coords) for r in res])
        res_segment_lengths = self._ceil_to_size(self.block_size, res_lengths)

        segment_ends = res_segment_lengths.cumsum()
        segment_starts = numpy.empty_like(segment_ends)
        segment_starts[0] = 0
        segment_starts[1:] = segment_ends[:-1]

        res_aidx = segment_starts

        # Allocate the coordinate system and attach residues
        buffer_size = segment_ends[-1]
        cbuff = numpy.full((buffer_size, 3), numpy.nan)

        attached_res = [
            r.attach_to(cbuff[start:start + len(r.coords)])
            for r, start in zip(res, res_aidx)
        ]

        ### Generate atom metadata

        atom_metadata = numpy.empty(buffer_size, self.atom_metadata_dtype)
        atom_metadata["atom_index"] = numpy.arange(len(atom_metadata))
        atom_metadata["residue_index"] = None

        for ri, (rs, r) in enumerate(zip(res_aidx, res)):
            rt = r.residue_type
            residue_block = atom_metadata[rs:rs + len(rt.atoms)]
            residue_block["residue_name"] = rt.name
            residue_block["atom_name"] = [a.name for a in rt.atoms]
            residue_block["atom_type"] = [a.atom_type for a in rt.atoms]
            residue_block["residue_index"] = ri

        ### Index residue connectivity
        # Generate a table of residue connections, with "from" and "to" entries
        # for *both* directions Just a linear set of connections for now
        residue_connections = pandas.DataFrame.from_records(
            [(i, "up", i + 1, "down") for i in range(len(res) - 1)],
            columns=pandas.MultiIndex.from_tuples([
                ("from", "resi"),
                ("from", "cname"),
                ("to", "resi"),
                ("to", "cname"),
            ])
        )
        connection_index = pandas.concat((
                residue_connections,
                residue_connections.rename(
                    columns={"from": "to", "to": "from"}
                )),
            ignore_index=True,
        ) # yapf: disable

        # Generate an index of all the connection atoms in the system,
        # resolving the internal and global index of the connection atoms
        connection_atoms = pandas.DataFrame.from_records([
                (ri, cname, c_aidx, c_aidx + r_g_aidx)
                for (ri, (r_g_aidx, r)) in enumerate(zip(res_aidx, res))
                for cname, c_aidx in r.residue_type.connection_to_idx.items()
            ],
            columns=["resi", "cname", "internal_aidx", "aidx"]
        ) # yapf: disable

        # Merge against the connection table to generate a connection entry
        # with the residue index, the connection name, the local atom index,
        # and the global atom index for the connection in the columns:
        #
        # cname  resi  internal_aidx  aidx
        from_connections = pandas.merge(
            connection_index["from"],
            connection_atoms,
        )
        to_connections = pandas.merge(
            connection_index["to"],
            connection_atoms,
        )

        for c in from_connections.columns:
            connection_index["from", c] = from_connections[c]
        for c in to_connections.columns:
            connection_index["to", c] = to_connections[c]

        ### Generate the bond graph

        # Offset the internal bond graph by the residue start idx
        intra_res_bonds = numpy.concatenate([
            r.residue_type.bond_indicies + start
            for start, r in zip(segment_starts, res)
        ])

        # Join the connection global atom indices
        inter_res_bonds = numpy.vstack([
            from_connections["aidx"].values, to_connections["aidx"].values
        ]).T

        bonds = numpy.concatenate([
            intra_res_bonds,
            inter_res_bonds,
        ])

        ### Generate dihedral metadata for all named torsions

        # Unpack all the residue type torsion entries, and tag with the
        # source residue index
        torsion_entries = [
            dict(
                residue_index=ri,
                **torsion_entry,
            )
            for ri, r in enumerate(res)
            for torsion_entry in cattr.unstructure(r.residue_type.torsions)
        ]

        # Generate a lookup from residue/connection to connected residue
        connection_lookup = pandas.concat(
            (
                pandas.DataFrame( # All the named connections
                    dict(
                        residue_index=connection_index["from", "resi"],
                        cname=connection_index["from", "cname"],
                        to_residue=connection_index["to", "resi"],
                    )
                ),
                pandas.DataFrame( # Loop-back to self for unamed connections
                    dict(
                        cname=None,
                        residue_index=numpy.arange(len(res)),
                        to_residue=numpy.arange(len(res)),
                    )
                ),
            ),
            ignore_index=True,
        )

        # Generate a lookup from residue index and atom name to global atom index.
        atom_lookup = pandas.DataFrame(
            dict(
                residue_index=atom_metadata["residue_index"],
                atom_name=atom_metadata["atom_name"],
                atom_index=numpy.arange(len(atom_metadata)),
            )
        )
        atom_lookup = atom_lookup[~atom_lookup.residue_index.isna()]

        # Left merge the residue/connection name into a target residue, and
        # then the target residue and atom name into a global atom index
        # for all atoms in the torsion (a, b, c, d).

        # This yields a global torsion table every torsion, the torsion name,
        # and the associated global atom indices.
        torsion_index = toolz.reduce(toolz.curry(pandas.merge)(how="left", copy=False), (
            pandas.io.json.json_normalize(torsion_entries),
            connection_lookup.rename(
                columns={"cname": "a.connection", "to_residue": "a.residue"}),
            atom_lookup.rename(
                columns={"residue_index": "a.residue", "atom_name": "a.atom", "atom_index": "a.atom_index"}),
            connection_lookup.rename(
                columns={"cname": "b.connection", "to_residue": "b.residue"}),
            atom_lookup.rename(
                columns={"residue_index": "b.residue", "atom_name": "b.atom", "atom_index": "b.atom_index"}),
            connection_lookup.rename(
                columns={"cname": "c.connection", "to_residue": "c.residue"}),
            atom_lookup.rename(
                columns={"residue_index": "c.residue", "atom_name": "c.atom", "atom_index": "c.atom_index"}),
            connection_lookup.rename(
                columns={"cname": "d.connection", "to_residue": "d.residue"}),
            atom_lookup.rename(
                columns={"residue_index": "d.residue", "atom_name": "d.atom", "atom_index": "d.atom_index"}),
        )).sort_index("columns") # yapf: disable

        self.residues = attached_res
        self.res_start_ind = segment_starts
        self.system_size = buffer_size

        self.coords = cbuff
        self.atom_metadata = atom_metadata
        self.atom_metadata.flags.writeable = False

        #TODO Convert to numpy structured arrays?
        self.connection_index = connection_index.sort_index("columns")
        self.torsion_index = torsion_index.sort_index("columns")

        self.bonds = bonds

        self.validate()

        return self
