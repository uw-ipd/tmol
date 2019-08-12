from typing import Optional, Tuple

import torch
import numpy

from ..database import ParamDB
from ..device import TorchDevice
from ..bonded_atom import BondedAtomScoreGraph, IndexedBonds
from ..score_components import ScoreComponentClasses, IntraScore
from ..score_graph import score_graph

from tmol.database import ParameterDatabase
from tmol.database.scoring import CartBondedDatabase
from .identification import CartBondedIdentification
from .params import CartBondedParamResolver
from .script_modules import CartBondedModule


from tmol.utility.reactive import reactive_attrs, reactive_property
from tmol.types.functional import validate_args

from tmol.types.array import NDArray

from tmol.types.torch import Tensor


@reactive_attrs
class CartBondedIntraScore(IntraScore):
    @reactive_property
    # @validate_args
    def cartbonded_score(target):
        return target.cartbonded_module(
            target.coords,
            target.cartbonded_lengths,
            target.cartbonded_angles,
            target.cartbonded_torsions,
            target.cartbonded_impropers,
            target.cartbonded_hxltorsions,
        )

    @reactive_property
    def total_cartbonded_length(cartbonded_score):
        return cartbonded_score[:, 0]

    @reactive_property
    def total_cartbonded_angle(cartbonded_score):
        return cartbonded_score[:, 1]

    @reactive_property
    def total_cartbonded_torsion(cartbonded_score):
        return cartbonded_score[:, 2]

    @reactive_property
    def total_cartbonded_improper(cartbonded_score):
        return cartbonded_score[:, 3]

    @reactive_property
    def total_cartbonded_hxltorsion(cartbonded_score):
        return cartbonded_score[:, 4]


@validate_args
def select_names_from_indices(
    res_names: NDArray(object)[:, :],
    atom_names: NDArray(object)[:, :],
    atom_indices: NDArray(int)[:, :, :],
    atom_for_resid: int,
) -> Tuple[NDArray(object)[:, :], ...]:
    resnames = numpy.full(atom_indices.shape[0:2], None, dtype=object)
    atnames = []
    for i in range(atom_indices.shape[2]):
        atnames.append(numpy.full_like(resnames, None))
    for i in range(atom_indices.shape[0]):
        nreal = numpy.sum(atom_indices[i, :, 0] >= 0)
        resnames[i, :nreal] = res_names[i, atom_indices[i, :nreal, atom_for_resid]]
        for j in range(atom_indices.shape[2]):
            atnames[j][i, :nreal] = atom_names[i, atom_indices[i, :nreal, j]]
    return (resnames,) + tuple(atnames)

    # resnames = numpy.full_like(bondlength_atom_indices, None, dtype=object)
    # at1names = numpy.full_like(bondlength_atom_indices, None, dtype=object)
    # at2names = numpy.full_like(bondlength_atom_indices, None, dtype=object)
    #
    # nstacks = bondlength_atom_indices.shape[0]
    # for i in range(nstacks):
    #     nreal = numpy.sum(bondlength_atom_indices[i] > 0)
    #     resnames[i, :nreal] = res_names[i, bondlength_atom_indices[i, :nreal]]
    #     at1names[i, :nreal] = atom_names[i, bondlength_atom_indices[i, :nreal, 0]]
    #     at2names[i, :nreal] = atom_names[i, bondlength_atom_indices[i, :nreal, 1]]


@validate_args
def remove_undefined_indices(
    atom_inds: NDArray(numpy.int64)[:, :, :],
    param_inds: NDArray(numpy.int64)[:, :],
    device: torch.device,
) -> Tensor(torch.long)[:, :, :]:
    """Prune out the below-zero entries from the param inds
    tensor and concatenate the remaining entries with the
    corresponding entries from the atom-inds tensor. The
    atom_inds tensor should be
    [ nstacks x nentries x natoms-per-entry ].
    The param_inds tensor should be
    [ nstacks x nentries ].
    The output tensor will be
    [ nstacks x max-non-zero-params-per-stack x natoms-per-entry+1 ]
    where a sentinel value of -9999 will be present
    if either the param- or the atom index represents
    a non-existent atom set."""

    assert atom_inds.shape[0] == param_inds.shape[0]
    assert atom_inds.shape[1] == param_inds.shape[1]

    nstacks = atom_inds.shape[0]
    stack_keep = []
    for i in range(nstacks):
        keep = param_inds[i] >= 0
        stack_keep.append(keep)
    max_keep = max(numpy.sum(keep) for keep in stack_keep)
    cb_inds = torch.full(
        (nstacks, max_keep, atom_inds.shape[2] + 1), -9999, dtype=torch.int64
    )
    for i in range(nstacks):
        keep = stack_keep[i]
        nkeep = numpy.sum(keep)
        cb_inds[i, :nkeep, :-1] = torch.tensor(atom_inds[i, keep], dtype=torch.int64)
        cb_inds[i, :nkeep, -1] = torch.tensor(param_inds[i, keep], dtype=torch.int64)

    return cb_inds.to(device=device)


@score_graph
class CartBondedScoreGraph(BondedAtomScoreGraph, ParamDB, TorchDevice):
    """Compute graph for the CartBonded term.
    """

    total_score_components = [
        ScoreComponentClasses(
            "cartbonded_length",
            intra_container=CartBondedIntraScore,
            inter_container=None,
        ),
        ScoreComponentClasses(
            "cartbonded_angle",
            intra_container=CartBondedIntraScore,
            inter_container=None,
        ),
        ScoreComponentClasses(
            "cartbonded_torsion",
            intra_container=CartBondedIntraScore,
            inter_container=None,
        ),
        ScoreComponentClasses(
            "cartbonded_improper",
            intra_container=CartBondedIntraScore,
            inter_container=None,
        ),
        ScoreComponentClasses(
            "cartbonded_hxltorsion",
            intra_container=CartBondedIntraScore,
            inter_container=None,
        ),
    ]

    @staticmethod
    def factory_for(
        val,
        parameter_database: ParameterDatabase,
        device: torch.device,
        cartbonded_database: Optional[CartBondedDatabase] = None,
        **_,
    ):
        """Overridable clone-constructor.
        """
        if cartbonded_database is None:
            if getattr(val, "cartbonded_database", None):
                cartbonded_database = val.cartbonded_database
            else:
                cartbonded_database = parameter_database.scoring.cartbonded

        return dict(cartbonded_database=cartbonded_database)

    cartbonded_database: CartBondedDatabase

    @reactive_property
    def cartbonded_param_resolver(
        cartbonded_database: CartBondedDatabase, device: torch.device
    ) -> CartBondedParamResolver:
        "cartbonded tuple resolver"
        return CartBondedParamResolver.from_database(cartbonded_database, device)

    @reactive_property
    def cartbonded_param_identifier(
        cartbonded_database: CartBondedDatabase, indexed_bonds: IndexedBonds
    ) -> CartBondedIdentification:
        return CartBondedIdentification.setup(
            cartbonded_database=cartbonded_database, indexed_bonds=indexed_bonds
        )

    @reactive_property
    def cartbonded_module(
        cartbonded_param_resolver: CartBondedParamResolver,
    ) -> CartBondedModule:
        return CartBondedModule(cartbonded_param_resolver)

    @reactive_property
    def cartbonded_lengths(
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :, 3]:

        # combine resolved atom indices and bondlength indices
        bondlength_atom_indices = cartbonded_param_identifier.lengths

        res, at1, at2 = select_names_from_indices(
            res_names, atom_names, bondlength_atom_indices, atom_for_resid=0
        )

        bondlength_indices = cartbonded_param_resolver.resolve_lengths(res, at1, at2)

        return remove_undefined_indices(
            bondlength_atom_indices,
            bondlength_indices,
            cartbonded_param_resolver.device,
        )

    @reactive_property
    def cartbonded_angles(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :, 4]:
        # combine resolved atom indices and bondangle indices
        bondangle_atom_indices = cartbonded_param_identifier.angles

        res, at1, at2, at3 = select_names_from_indices(
            res_names, atom_names, bondangle_atom_indices, atom_for_resid=1
        )

        bondangle_indices = cartbonded_param_resolver.resolve_angles(res, at1, at2, at3)

        return remove_undefined_indices(
            bondangle_atom_indices, bondangle_indices, cartbonded_param_resolver.device
        )

    @reactive_property
    def cartbonded_torsions(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :, 5]:
        # combine resolved atom indices and bondangle indices
        torsion_atom_indices = cartbonded_param_identifier.torsions
        # use atm2 for resid
        res, at1, at2, at3, at4 = select_names_from_indices(
            res_names, atom_names, torsion_atom_indices, atom_for_resid=1
        )
        torsion_indices = cartbonded_param_resolver.resolve_torsions(
            res, at1, at2, at3, at4
        )
        return remove_undefined_indices(
            torsion_atom_indices, torsion_indices, cartbonded_param_resolver.device
        )

    @reactive_property
    def cartbonded_impropers(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :, 5]:
        # combine resolved atom indices and bondangle indices
        improper_atom_indices = cartbonded_param_identifier.impropers
        # use atm3 for resid
        res, at1, at2, at3, at4 = select_names_from_indices(
            res_names, atom_names, improper_atom_indices, atom_for_resid=2
        )
        improper_indices = cartbonded_param_resolver.resolve_impropers(
            res, at1, at2, at3, at4
        )

        return remove_undefined_indices(
            improper_atom_indices,
            improper_indices,
            device=cartbonded_param_resolver.device,
        )

    @reactive_property
    def cartbonded_hxltorsions(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :, 5]:
        # same identification as regular torsions, but resolved against a different DB
        hxltorsion_atom_indices = cartbonded_param_identifier.torsions
        res, at1, at2, at3, at4 = select_names_from_indices(
            res_names, atom_names, hxltorsion_atom_indices, atom_for_resid=2
        )
        hxltorsion_indices = cartbonded_param_resolver.resolve_hxltorsions(
            res, at1, at2, at3, at4
        )

        return remove_undefined_indices(
            hxltorsion_atom_indices,
            hxltorsion_indices,
            cartbonded_param_resolver.device,
        )
