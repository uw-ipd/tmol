import torch
import attr
import cattr
import numpy
import pandas

import scipy.sparse.csgraph

from tmol.types.attrs import ValidateAttrs
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.system.packed import PackedResidueSystem
from tmol.system.restypes import ResidueType


@attr.s(auto_attribs=True, frozen=True)
class OneRestypeRotamerSet(ValidateAttrs):
    rotamer_coords: Tensor(torch.float)[:, :, 3]
    restype: ResidueType

    @classmethod
    def for_seqpos(cls, system: PackedResidueSystem, seqpos: int, target: ResidueType):
        pass


@attr.s(frozen=True)
class AASidechainBuilder:
    # look, I'm just going to figure out the kinematics of building rotamers.
    # the question of where data lives and how it flows into the right place
    # is open and one we'll need to explore, but if we're just going to prototype
    # the packer as it currently exists, then we don't really need to tackle
    # that first.
    #
    # this is unquestionably the wrong place to encode "what atoms are backbone atoms"
    # and it's unquestionably protein centric, but that's ok for today.
    backbone_atoms = ["N", "CA", "C", "O"]

    restype: ResidueType = attr.ib()
    is_backbone_atom: Tensor(int)[:] = attr.ib()
    backbone_atom_inds: Tensor(int)[:] = attr.ib()
    icoor_for_ats: Tensor(int)[:, 3] = attr.ib()
    chi_spins_dof: Tensor(int)[:, 3] = attr.ib()

    @classmethod
    def from_restype(cls, restype: ResidueType):
        natoms = len(restype.atoms)
        nchi = sum(1 for tor in restype.torsions if "chi" in tor.name)
        bonds = numpy.zeros((natoms, natoms), dtype=int)
        for ai, bi in restype.bond_indices:
            bonds[ai, bi] = 1
        ca_ind = restype.atom_to_idx["CA"]
        dfs_ind, dfs_parent = scipy.sparse.csgraph.depth_first_order(bonds, ca_ind)

        is_backbone_atom = numpy.zeros(natoms, dtype=int)
        backbone_atom_inds = numpy.zeros(
            len(AASidechainBuilder.backbone_atoms), dtype=int
        )
        for i, at in enumerate(AASidechainBuilder.backbone_atoms):
            ati = restype.atom_to_idx[at]
            is_backbone_atom[ati] = 1
            backbone_atom_inds[i] = ati

        ideal_dofs = numpy.zeros((natoms, 3), dtype=float)
        # dofs in order:
        # dihedral from great-grandparent
        # angle from grand parent
        # bond length from parent

        icoor_df = pandas.DataFrame(list(cattr.unstructure(restype.icoors))).set_index(
            "name"
        )
        names = [atom.name for atom in restype.atoms]

        par = numpy.array(
            [restype.atom_to_idx[p] for p in icoor_df.loc[names]["parent"]]
        )
        gpar = numpy.array(
            [restype.atom_to_idx[gp] for gp in icoor_df.loc[names]["grand_parent"]]
        )
        ggpar = numpy.array(
            [
                (restype.atom_to_idx[ggp] if ggp in names else -1)
                for ggp in icoor_df.loc[names]["great_grand_parent"]
            ]
        )

        # chi spins the torsion for an atom if it is the last atom
        # atom for that chi or if its great-grand-parent is the last
        # atom for the chi and its parent is the second-to-last atom
        # for the chi. In this latter case, the a atom's ggp defines
        # an improper torsion.
        last_chi_ats = numpy.array(
            [
                restype.atom_to_idx[tor.d.atom]
                for tor in restype.torsions
                if "chi" in tor.name
            ],
            dtype=int,
        )
        second_to_last_chi_ats = numpy.array(
            [
                restype.atom_to_idx[tor.c.atom]
                for tor in restype.torsions
                if "chi" in tor.name
            ],
            dtype=int,
        )
        chi_spins_dof = numpy.full(natoms, -1, dtype=int)
        chi_spins_dof[last_chi_ats] = numpy.arange(len(last_chi_ats), dtype=int)
        for i, at in enumerate(names):
            for j, (last, sectolast) in enumerate(
                zip(last_chi_ats, second_to_last_chi_ats)
            ):
                if sectolast == par[i] and chi_spins_dof[ggpar[i]] == j:
                    chi_spins_dof[i] = j

        icoor_for_ats = numpy.array(
            [
                (row["phi"], row["theta"], row["d"])
                for _, row in icoor_df.loc[names].iterrows()
            ],
            dtype=float,
        )

        icoor_for_ats[chi_spins_dof != -1, 0] -= icoor_for_ats[
            last_chi_ats[chi_spins_dof[chi_spins_dof != -1]], 0
        ]

        return cls(
            restype=restype,
            is_backbone_atom=is_backbone_atom,
            backbone_atom_inds=backbone_atom_inds,
            icoor_for_ats=icoor_for_ats,
            chi_spins_dof=chi_spins_dof,
        )
