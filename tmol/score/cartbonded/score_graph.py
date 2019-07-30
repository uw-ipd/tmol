from typing import Optional

import torch

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

from tmol.types.array import NDArray

from tmol.types.torch import Tensor


@reactive_attrs
class CartBondedIntraScore(IntraScore):
    @reactive_property
    # @validate_args
    def cartbonded_score(target):
        return target.cartbonded_module(
            target.coords[0, ...],
            target.cartbonded_lengths,
            target.cartbonded_angles,
            target.cartbonded_torsions,
            target.cartbonded_impropers,
            target.cartbonded_hxltorsions,
        )

    @reactive_property
    def total_cartbonded_length(cartbonded_score):
        return cartbonded_score[None, 0]

    @reactive_property
    def total_cartbonded_angle(cartbonded_score):
        return cartbonded_score[None, 1]

    @reactive_property
    def total_cartbonded_torsion(cartbonded_score):
        return cartbonded_score[None, 2]

    @reactive_property
    def total_cartbonded_improper(cartbonded_score):
        return cartbonded_score[None, 3]

    @reactive_property
    def total_cartbonded_hxltorsion(cartbonded_score):
        return cartbonded_score[None, 4]


@validate_args
def remove_undefined_indices(
    atom_inds: NDArray(numpy.int64)[:, :, :],
    param_inds: NDArray(numpy.int64)[:, :],
    device: torch.device,
) -> torch.Tensor(torch.long)[:, :]:
    assert atom_inds.shape[0] == param_inds.shape[0]
    assert atom_inds.shape[1] == param_inds.shape[1]

    nstacks = atom_inds.shape[0]
    stack_keep = []
    for i in range(nstacks):
        keep = param_inds[i] >= 0
        stack_keep.append(keep)
    max_keep = max(numpy.sum(keep) for keep in stack_keep)
    cb_inds = torch.tensor(
        (nstacks, max_keep, atom_inds.shape[2] + 1), dtype=torch.int64
    )
    for i in range(nstacks):
        keep = stack_keep[i]
        nkeep = numpy.sum(keep)
        cb_inds[i, :nkeep, :-1] = atom_inds[i, keep]
        cb_inds[i, :nkeep, -1] = param_inds[i, keep]

    return cb_inds.to(device=cartbonded_param_resolver.device)


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
    ) -> Tensor(torch.int64)[:, :]:
        # combine resolved atom indices and bondlength indices
        bondlength_atom_indices = cartbonded_param_identifier.lengths
        resnames = numpy.full_like(bondlength_atom_indices, None, dtype=object)
        at1names = numpy.full_like(bondlength_atom_indices, None, dtype=object)
        at2names = numpy.full_like(bondlength_atom_indices, None, dtype=object)

        nstacks = bondlength_atom_indices.shape[0]
        for i in range(nstacks):
            nreal = numpy.sum(bondlength_atom_indices[i] > 0)
            resnames[i, :nreal] = res_names[i, bondlength_atom_indices[i, :nreal]]
            at1names[i, :nreal] = atom_names[i, bondlength_atom_indices[i, :nreal, 0]]
            at2names[i, :nreal] = atom_names[i, bondlength_atom_indices[i, :nreal, 1]]

        bondlength_indices = cartbonded_param_resolver.resolve_lengths(
            resnames, at1names, at2names
        )

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
    ) -> Tensor(torch.int64)[:, :]:
        # combine resolved atom indices and bondangle indices
        bondangle_atom_indices = cartbonded_param_identifier.angles
        resnames = numpy.full_like(bondangle_atom_indices, None, dtype=object)
        at1names = numpy.full_like(bondangle_atom_indices, None, dtype=object)
        at2names = numpy.full_like(bondangle_atom_indices, None, dtype=object)
        at3names = numpy.full_like(bondangle_atom_indices, None, dtype=object)

        nstacks = bondlength_atom_indices.shape[0]
        for i in range(nstacks):
            nreal = numpy.sum(bondlength_atom_indices[i] > 0)
            resnames[i, :nreal] = res_names[i, bondangle_atom_indices[i, :nreal]]
            at1names[i, :nreal] = atom_names[i, bondangle_atom_indices[i, :nreal, 0]]
            at2names[i, :nreal] = atom_names[i, bondangle_atom_indices[i, :nreal, 1]]
            at3names[i, :nreal] = atom_names[i, bondangle_atom_indices[i, :nreal, 2]]

        bondangle_indices = cartbonded_param_resolver.resolve_angles(
            resnames, at1names, at2names, at3names
        )

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
    ) -> Tensor(torch.int64)[:, :]:
        # combine resolved atom indices and bondangle indices
        torsion_atom_indices = cartbonded_param_identifier.torsions
        torsion_indices = cartbonded_param_resolver.resolve_torsions(
            res_names[0, torsion_atom_indices[:, 1]],  # use atm2 for resid
            atom_names[0, torsion_atom_indices[:, 0]],
            atom_names[0, torsion_atom_indices[:, 1]],
            atom_names[0, torsion_atom_indices[:, 2]],
            atom_names[0, torsion_atom_indices[:, 3]],
        )

        # remove undefined indices
        torsion_defined = torsion_indices != -1

        cbt = torch.cat(
            [
                torch.tensor(torsion_atom_indices[torsion_defined]),
                torch.tensor(torsion_indices[torsion_defined, None]),
            ],
            dim=1,
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int64)

        return cbt

    @reactive_property
    def cartbonded_impropers(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :]:
        # combine resolved atom indices and bondangle indices
        improper_atom_indices = cartbonded_param_identifier.impropers
        improper_indices = cartbonded_param_resolver.resolve_impropers(
            res_names[0, improper_atom_indices[:, 2]],  # use atm3 for resid
            atom_names[0, improper_atom_indices[:, 0]],
            atom_names[0, improper_atom_indices[:, 1]],
            atom_names[0, improper_atom_indices[:, 2]],
            atom_names[0, improper_atom_indices[:, 3]],
        )

        # remove undefined indices
        improper_defined = improper_indices != -1

        cbi = torch.cat(
            [
                torch.tensor(improper_atom_indices[improper_defined]),
                torch.tensor(improper_indices[improper_defined, None]),
            ],
            dim=1,
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int64)

        return cbi

    @reactive_property
    def cartbonded_hxltorsions(
        bonds: NDArray(int)[:, 3],
        res_names: NDArray(object)[...],
        atom_names: NDArray(object)[...],
        cartbonded_param_resolver: CartBondedParamResolver,
        cartbonded_param_identifier: CartBondedIdentification,
    ) -> Tensor(torch.int64)[:, :]:
        # same identification as regular torsions, but resolved against a different DB
        hxltorsion_atom_indices = cartbonded_param_identifier.torsions
        hxltorsion_indices = cartbonded_param_resolver.resolve_hxltorsions(
            res_names[0, hxltorsion_atom_indices[:, 2]],  # use atm3 for resid
            atom_names[0, hxltorsion_atom_indices[:, 0]],
            atom_names[0, hxltorsion_atom_indices[:, 1]],
            atom_names[0, hxltorsion_atom_indices[:, 2]],
            atom_names[0, hxltorsion_atom_indices[:, 3]],
        )

        # remove undefined indices
        hxltorsion_defined = hxltorsion_indices != -1

        cbht = torch.cat(
            [
                torch.tensor(hxltorsion_atom_indices[hxltorsion_defined]),
                torch.tensor(hxltorsion_indices[hxltorsion_defined, None]),
            ],
            dim=1,
        ).to(device=cartbonded_param_resolver.device, dtype=torch.int64)

        return cbht
