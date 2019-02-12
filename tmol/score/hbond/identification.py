import torch
import attr

from tmol.types.functional import convert_args
from tmol.types.attrs import ValidateAttrs
from tmol.types.array import NDArray

import numpy

from tmol.database.chemical import ChemicalDatabase
from tmol.database.scoring import HBondDatabase
from tmol.score.chemical_database import AtomTypeParamResolver
from ..bonded_atom import IndexedBonds

from tmol.utility.cpp_extension import load, modulename, relpaths

acceptor_dtype = numpy.dtype(
    [("a", int), ("b", int), ("b0", int), ("acceptor_type", object)]
)

donor_dtype = numpy.dtype([("d", int), ("h", int), ("donor_type", object)])


@attr.s(frozen=True, slots=True, auto_attribs=True)
class HBondElementAnalysis(ValidateAttrs):
    donors: NDArray(donor_dtype)[:]
    acceptors: NDArray(acceptor_dtype)[:]

    @classmethod
    @convert_args
    def setup(
        cls,
        hbond_database: HBondDatabase,
        atom_types: NDArray(object)[:],
        atom_is_acceptor: NDArray(bool)[:],
        atom_acceptor_hybridization: NDArray(int)[:],
        atom_is_donor: NDArray(bool)[:],
        atom_is_hydrogen: NDArray(bool)[:],
        bonds: NDArray(int)[:, 2],
    ):

        compiled = load(
            modulename(__name__), relpaths(__file__, "identification.pybind.cc")
        )

        # Only access bonds through index.
        bonds = IndexedBonds.from_bonds(bonds, minlength=len(atom_types))

        # Filter donors/acceptors to those with type definitions in hbond
        # database, this logic should likely be moved to parameter resolution.
        atom_type_acceptor_type = {
            p.a: p.acceptor_type for p in hbond_database.acceptor_atom_types
        }
        atom_type_donor_type = {
            p.d: p.donor_type for p in hbond_database.donor_atom_types
        }

        atom_acceptor_type = numpy.array(
            [atom_type_acceptor_type.get(at, None) for at in atom_types]
        )
        atom_donor_type = numpy.array(
            [atom_type_donor_type.get(at, None) for at in atom_types]
        )

        # Get the acceptor indicies and allocate base idx buffers
        A_idx = numpy.flatnonzero(
            atom_is_acceptor & atom_acceptor_type.astype(bool)  # None->False
        )
        B_idx = numpy.empty_like(A_idx)
        B0_idx = numpy.empty_like(A_idx)

        # Yeeeehaw
        compiled.id_acceptor_bases(
            torch.from_numpy(A_idx),
            torch.from_numpy(B_idx),
            torch.from_numpy(B0_idx),
            torch.from_numpy(atom_acceptor_hybridization),
            torch.from_numpy(atom_is_hydrogen.astype(numpy.ubyte)),
            bonds,
        )

        assert not numpy.any(B_idx == -1), "Invalid acceptor atom type."

        acceptors = numpy.empty(A_idx.shape, dtype=acceptor_dtype)
        acceptors["a"] = A_idx
        acceptors["b"] = B_idx
        acceptors["b0"] = B0_idx
        acceptors["acceptor_type"] = atom_acceptor_type[acceptors["a"]]

        # Identify donor groups via donor-hydrogen bonds.
        atom_type_donor_type = {
            p.d: p.donor_type for p in hbond_database.donor_atom_types
        }

        donor_pair_idx = bonds.bonds.numpy()[
            atom_is_donor[bonds.bonds[:, 0]]
            & atom_donor_type.astype(bool)[bonds.bonds[:, 0]]  # None -> False
            & atom_is_hydrogen[bonds.bonds[:, 1]]
        ]

        donors = numpy.empty(donor_pair_idx.shape[0], dtype=donor_dtype)
        donors["d"] = donor_pair_idx[:, 0]
        donors["h"] = donor_pair_idx[:, 1]
        donors["donor_type"] = atom_donor_type[donors["d"]]

        return cls(donors=donors, acceptors=acceptors)

    @classmethod
    def setup_from_database(
        cls,
        chemical_database: ChemicalDatabase,
        hbond_database: HBondDatabase,
        atom_types: NDArray(object)[:],
        bonds: NDArray(int)[:, 2],
    ):

        atom_resolver = AtomTypeParamResolver.from_database(
            chemical_database, torch.device("cpu")
        )
        atom_type_idx = atom_resolver.type_idx(atom_types)
        atom_type_params = atom_resolver.params[atom_type_idx]

        return cls.setup(
            hbond_database=hbond_database,
            atom_types=atom_types,
            atom_is_acceptor=atom_type_params.is_acceptor.numpy(),
            atom_acceptor_hybridization=atom_type_params.acceptor_hybridization.numpy(),
            atom_is_donor=atom_type_params.is_donor.numpy(),
            atom_is_hydrogen=atom_type_params.is_hydrogen.numpy(),
            bonds=bonds,
        )
