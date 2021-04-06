import attr
from attrs_strict import type_validator
from typing import Set, Type, Optional, Tuple
import torch
import numpy
from functools import singledispatch

from tmol.database.scoring import CartBondedDatabase

from tmol.score.cartbonded.identification import CartBondedIdentification
from tmol.score.cartbonded.params import CartBondedParamResolver
from tmol.score.cartbonded.script_modules import CartBondedModule

from tmol.score.modules.bases import ScoreSystem, ScoreModule, ScoreMethod
from tmol.score.modules.device import TorchDevice
from tmol.score.modules.database import ParamDB
from tmol.score.modules.bonded_atom import BondedAtoms

from tmol.types.array import NDArray
from tmol.types.functional import validate_args
from tmol.types.torch import Tensor


@attr.s(slots=True, auto_attribs=True, kw_only=True, frozen=True)
class CartBondedParameters(ScoreModule):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {BondedAtoms, ParamDB, TorchDevice}

    @staticmethod
    @singledispatch
    def build_for(
        val,
        system: ScoreSystem,
        *,
        cartbonded_database: Optional[CartBondedDatabase] = None,
        **_,
    ):
        """Override constructor.

        Create from provided `cartbonded_database``, otherwise from
        ``parameter_database.scoring.cartbonded``.
        """
        if cartbonded_database is None:
            cartbonded_database = ParamDB.get(
                system
            ).parameter_database.scoring.cartbonded

        return CartBondedParameters(
            system=system, cartbonded_database=cartbonded_database
        )

    cartbonded_database: CartBondedDatabase = attr.ib(validator=type_validator())
    cartbonded_param_resolver: CartBondedParamResolver = attr.ib(init=False)
    cartbonded_param_identifier: CartBondedIdentification = attr.ib(init=False)

    cartbonded_lengths: Tensor[torch.int64][:, :, 3] = attr.ib(init=False)
    cartbonded_angles: Tensor[torch.int64][:, :, 4] = attr.ib(init=False)
    cartbonded_torsions: Tensor[torch.int64][:, :, 5] = attr.ib(init=False)
    cartbonded_impropers: Tensor[torch.int64][:, :, 5] = attr.ib(init=False)
    cartbonded_hxltorsions: Tensor[torch.int64][:, :, 5] = attr.ib(init=False)

    @cartbonded_param_resolver.default
    def _init_cartbonded_param_resolver(self) -> CartBondedParamResolver:
        # torch.device for param resolver is inherited from chemical db
        return CartBondedParamResolver.from_database(
            self.cartbonded_database, TorchDevice.get(self.system).device
        )

    @cartbonded_param_identifier.default
    def _init_cartbonded_param_identifier(self) -> CartBondedIdentification:
        return CartBondedIdentification.setup(
            cartbonded_database=self.cartbonded_database,
            indexed_bonds=BondedAtoms.get(self).indexed_bonds,
        )

    @cartbonded_lengths.default
    def _init_cartbonded_lengths(self,) -> Tensor[torch.int64][:, :, 3]:

        # combine resolved atom indices and bondlength indices
        bondlength_atom_indices = self.cartbonded_param_identifier.lengths

        res, at1, at2 = select_names_from_indices(
            BondedAtoms.get(self).res_names,
            BondedAtoms.get(self).atom_names,
            bondlength_atom_indices,
            atom_for_resid=0,
        )

        bondlength_indices = self.cartbonded_param_resolver.resolve_lengths(
            res, at1, at2
        )

        return remove_undefined_indices(
            bondlength_atom_indices,
            bondlength_indices,
            self.cartbonded_param_resolver.device,
        )

    @cartbonded_angles.default
    def _init_cartbonded_angles(self) -> Tensor[torch.int64][:, :, 4]:
        # combine resolved atom indices and bondangle indices
        bondangle_atom_indices = self.cartbonded_param_identifier.angles

        res, at1, at2, at3 = select_names_from_indices(
            BondedAtoms.get(self).res_names,
            BondedAtoms.get(self).atom_names,
            bondangle_atom_indices,
            atom_for_resid=1,
        )

        bondangle_indices = self.cartbonded_param_resolver.resolve_angles(
            res, at1, at2, at3
        )

        return remove_undefined_indices(
            bondangle_atom_indices,
            bondangle_indices,
            self.cartbonded_param_resolver.device,
        )

    @cartbonded_torsions.default
    def _init_cartbonded_torsions(self) -> Tensor[torch.int64][:, :, 5]:
        # combine resolved atom indices and bondangle indices
        torsion_atom_indices = self.cartbonded_param_identifier.torsions
        # use atm2 for resid
        res, at1, at2, at3, at4 = select_names_from_indices(
            BondedAtoms.get(self).res_names,
            BondedAtoms.get(self).atom_names,
            torsion_atom_indices,
            atom_for_resid=1,
        )
        torsion_indices = self.cartbonded_param_resolver.resolve_torsions(
            res, at1, at2, at3, at4
        )
        return remove_undefined_indices(
            torsion_atom_indices, torsion_indices, self.cartbonded_param_resolver.device
        )

    @cartbonded_impropers.default
    def _init_cartbonded_impropers(self) -> Tensor[torch.int64][:, :, 5]:
        # combine resolved atom indices and bondangle indices
        improper_atom_indices = self.cartbonded_param_identifier.impropers
        # use atm3 for resid
        res, at1, at2, at3, at4 = select_names_from_indices(
            BondedAtoms.get(self).res_names,
            BondedAtoms.get(self).atom_names,
            improper_atom_indices,
            atom_for_resid=2,
        )
        improper_indices = self.cartbonded_param_resolver.resolve_impropers(
            res, at1, at2, at3, at4
        )

        return remove_undefined_indices(
            improper_atom_indices,
            improper_indices,
            device=self.cartbonded_param_resolver.device,
        )

    @cartbonded_hxltorsions.default
    def _init_cartbonded_hxltorsions(self) -> Tensor[torch.int64][:, :, 5]:
        # same identification as regular torsions, but resolved against a different DB
        hxltorsion_atom_indices = self.cartbonded_param_identifier.torsions
        res, at1, at2, at3, at4 = select_names_from_indices(
            BondedAtoms.get(self).res_names,
            BondedAtoms.get(self).atom_names,
            hxltorsion_atom_indices,
            atom_for_resid=2,
        )
        hxltorsion_indices = self.cartbonded_param_resolver.resolve_hxltorsions(
            res, at1, at2, at3, at4
        )

        return remove_undefined_indices(
            hxltorsion_atom_indices,
            hxltorsion_indices,
            self.cartbonded_param_resolver.device,
        )


@CartBondedParameters.build_for.register(ScoreSystem)
def _clone_for_score_system(
    old,
    system: ScoreSystem,
    *,
    cartbonded_database: Optional[CartBondedDatabase] = None,
    **_,
):
    """Override constructor.

        Create from ``val.cartbonded_database`` if possible, otherwise from
        ``parameter_database.scoring.cartbonded``.
        """
    if cartbonded_database is None:
        cartbonded_database = CartBondedParameters.get(old).cartbonded_database

    return CartBondedParameters(system=system, cartbonded_database=cartbonded_database)


@validate_args
def select_names_from_indices(
    res_names: NDArray[object][:, :],
    atom_names: NDArray[object][:, :],
    atom_indices: NDArray[int][:, :, :],
    atom_for_resid: int,
) -> Tuple[NDArray[object][:, :], ...]:
    resnames = numpy.full(atom_indices.shape[0:2], None, dtype=object)
    atnames = [numpy.full_like(resnames, None) for _ in range(atom_indices.shape[2])]
    real = atom_indices[:, :, 0] >= 0
    nz = numpy.nonzero(real)

    # masked assignment; nz[0] is the stack index, nz[1] is the torsion index
    resnames[real] = res_names[nz[0], atom_indices[nz[0], nz[1], atom_for_resid]]
    for i in range(atom_indices.shape[2]):
        atnames[i][real] = atom_names[nz[0], atom_indices[nz[0], nz[1], i]]

    return (resnames,) + tuple(atnames)


@validate_args
def remove_undefined_indices(
    atom_inds: NDArray[numpy.int64][:, :, :],
    param_inds: NDArray[numpy.int64][:, :],
    device: torch.device,
) -> Tensor[torch.long][:, :, :]:
    """Prune out the below-zero entries from the param inds
    tensor and concatenate the remaining entries with the
    corresponding entries from the atom-inds tensor. The
    atom_inds tensor should be
    [ nstacks x nentries x natoms-per-entry ].
    The param_inds tensor should be
    [ nstacks x nentries ].
    The output tensor will be
    [ nstacks x max-non-zero-params-per-stack x natoms-per-entry+1 ]
    where a sentinel value of -1 will be present
    if either the param- or the atom index represents
    a non-existent atom set.

    This code will "condense" an array with entries I'm not interested in
    into a smaller array so that fire up the minimum number of threads on
    the GPU that have no work to perform

    It will also "left shift" the valid entries so that the threads
    that do have no work do to are next to each other, thereby ensuring
    the highest warp coherency
    """

    assert atom_inds.shape[0] == param_inds.shape[0]
    assert atom_inds.shape[1] == param_inds.shape[1]

    # Find the non-negative set of parameter indices -- these correspond to
    # atom-tuples that should be scored, ie the real set.
    # Collapse these real atoms+parameters into the lowest entries
    # of an output tensor.

    nstacks = atom_inds.shape[0]
    real = torch.tensor(param_inds, dtype=torch.int32) >= 0
    nzreal = torch.nonzero(real)  # nz --> the indices of the real entries

    # how many for each stack should we keep?
    nkeep = torch.sum(real, dim=1).view((atom_inds.shape[0], 1))
    max_keep = torch.max(nkeep)
    cb_inds = torch.full(
        (nstacks, max_keep, atom_inds.shape[2] + 1), -1, dtype=torch.int64
    )

    # get the output-tensor indices for each stack that we should write to
    counts = torch.arange(max_keep, dtype=torch.int64).view((1, max_keep))
    lowinds = counts < nkeep
    nzlow = torch.nonzero(lowinds)

    cb_inds[nzlow[:, 0], nzlow[:, 1], :-1] = torch.tensor(atom_inds, dtype=torch.int64)[
        nzreal[:, 0], nzreal[:, 1]
    ]
    cb_inds[nzlow[:, 0], nzlow[:, 1], -1] = torch.tensor(param_inds, dtype=torch.int64)[
        nzreal[:, 0], nzreal[:, 1]
    ]

    return cb_inds.to(device=device)


@attr.s(slots=True, auto_attribs=True, kw_only=True)
class CartBondedScore(ScoreMethod):
    @staticmethod
    def depends_on() -> Set[Type[ScoreModule]]:
        return {CartBondedParameters}

    @staticmethod
    def build_for(val, system: ScoreSystem, **_) -> "CartBondedScore":
        return CartBondedScore(system=system)

    cartbonded_module: CartBondedModule = attr.ib(init=False)

    @cartbonded_module.default
    def _init_cartbonded_intra_module(self):
        return CartBondedModule(
            CartBondedParameters.get(self).cartbonded_param_resolver
        )

    def intra_forward(self, coords: torch.Tensor):
        return {
            "cartbonded": self.cartbonded_module(
                coords,
                CartBondedParameters.get(self).cartbonded_lengths,
                CartBondedParameters.get(self).cartbonded_angles,
                CartBondedParameters.get(self).cartbonded_torsions,
                CartBondedParameters.get(self).cartbonded_impropers,
                CartBondedParameters.get(self).cartbonded_hxltorsions,
            )
        }
