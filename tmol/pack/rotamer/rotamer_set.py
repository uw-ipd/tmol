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
from tmol.database import ChemicalDatabase


@attr.s(auto_attribs=True, frozen=True)
class OneRestypeRotamerSet(ValidateAttrs):
    rotamer_coords: Tensor(torch.float)[:, :, 3]
    restype: ResidueType

    @classmethod
    def for_seqpos(cls, system: PackedResidueSystem, seqpos: int, target: ResidueType):
        pass


@attr.s(auto_attribs=True, frozen=True)
class SingleSidechainBuilder:
    # Describe how to build the coordinates of the rotamers
    # for a single sidechain on a  single residue; each residue
    # may have more than one sidechain

    restype_name: str
    sidechain: int

    # how many atoms are built for this sidechain group?
    # it may be only a subset of the atoms for the residue
    natoms: int

    # mapping of atoms built for this rotamer to the indices in the ResidueType
    rotatom_2_resatom: Tensor(int)[:]
    # mapping of the atoms of the original ResidueType to the indices in this rotamer
    resatom_2_rotatom: Tensor(int)[:]

    # the bond matrix for the atoms that are built for this sidechain
    bonds: Tensor(int)[:, :]

    # which atoms of those built for this rotamer are backbone atoms?
    # i.e. which ones will take their coordinates from the existing residue
    is_backbone_atom: Tensor(int)[:]
    backbone_atom_inds: Tensor(int)[:]

    # sidechain root; in order:
    # 0) sidechain root atom ind (e.g. "CB")
    # 1) the backbone atom it's bonded to (e.g. "CA")
    # 2) the backone atom to orient the chi (e.g. "N")
    sidechain_root: Tensor(int)[3]

    # the order in which the atoms' coordinates should be built
    sidechain_dfs: Tensor(int)[:]

    # the phi/theta/d for all atoms built
    atom_icoors: Tensor(float)[:, 3]
    # the ancestor atoms whose coordinates are needed to build an atom
    atom_ancestors: Tensor(int)[:, 3]
    # the index of the chi dihedral that spins an atom, if any
    chi_that_spins_atom: Tensor(int)[:]

    @classmethod
    def determine_atoms_in_sidechain_group(
        cls, chem_db: ChemicalDatabase, restype: ResidueType, sidechain: int
    ):
        """The atoms that are included in a sidechain group are
        1) those that are listed as backbone atoms for this sidechain group.
        2) the hydrogen atoms bound to the backbone atoms,
        3) and the atoms reachable from a traversal from the sidechain root
        without traversing through the backbone atoms.
        Not all atoms in a residue will be included in a sidechain group.
        A residue may have multiple sidechain groups. Sidechain groups
        for a residue must not overlap.
        """

        natoms = len(restype.atoms)

        bonds = numpy.zeros((natoms, natoms), dtype=int)
        for ai, bi in restype.bond_indices:
            bonds[ai, bi] = 1

        bbats = restype.sidechain_building[sidechain].backbone_atoms
        keep = numpy.zeros(natoms, dtype=int)
        nonsc = []

        for bbat in [restype.atom_to_idx[at] for at in bbats]:
            keep[bbat] = 1
            nonsc.append(bbat)
            for other in range(natoms):
                if bonds[bbat, other]:
                    atype_name = restype.atoms[other].atom_type
                    atype = next(
                        filter(
                            lambda atype: atype.name == atype_name, chem_db.atom_types
                        )
                    )
                    if atype.element == "H":
                        keep[other] = 1
                        nonsc.append(other)
        sc_root_at = restype.sidechain_building[sidechain].root
        root = restype.atom_to_idx[sc_root_at]
        for other in range(natoms):
            if bonds[root, other] and keep[other]:
                bonds[root, other] = 0
                bonds[other, root] = 0
        dfs_ind, dfs_parent = scipy.sparse.csgraph.depth_first_order(bonds, root)
        for ind in dfs_ind:
            keep[ind] = 1
        return keep, nonsc, dfs_ind

    @classmethod
    def from_restype(
        cls, chem_db: ChemicalDatabase, restype: ResidueType, sidechain: int
    ):

        # which atoms should be included?
        # It should be everything reachable by a DFS from the sidechain
        # root
        keep, nonsc_kept, dfs_ind = cls.determine_atoms_in_sidechain_group(
            chem_db, restype, sidechain
        )

        natoms = len(restype.atoms)
        n_rotamer_atoms = sum(keep)
        rot2res = torch.zeros((n_rotamer_atoms,), dtype=torch.long)
        res2rot = torch.full((natoms,), -1, dtype=torch.long)
        count = 0
        for i, keep_i in enumerate(keep):
            if keep_i:
                res2rot[i] = count
                rot2res[count] = i
                count += 1

        bonds = numpy.zeros((natoms, natoms), dtype=int)
        for ai, bi in restype.bond_indices:
            bonds[ai, bi] = 1
        bonds = bonds[rot2res, :]
        bonds = bonds[:, rot2res]

        # sort the chi of the restypes to be ascending
        chi = sorted(
            [tor for tor in restype.torsions if "chi" in tor.name],
            key=lambda x: int(x.name.partition("chi")[2]),
        )
        nchi = len(chi)

        is_backbone_atom = torch.zeros((n_rotamer_atoms), dtype=torch.int)
        bbats = restype.sidechain_building[sidechain].backbone_atoms
        backbone_atom_inds = numpy.zeros(len(bbats), dtype=int)
        backbone_atom_inds = torch.tensor(
            [restype.atom_to_idx[at] for at in bbats], dtype=torch.long
        )
        backbone_atom_inds_rot = res2rot[backbone_atom_inds]
        is_backbone_atom[backbone_atom_inds] = 1

        ideal_dofs = torch.zeros((natoms, 3), dtype=torch.float)
        # dofs in order:
        # dihedral from great-grandparent
        # angle from grand parent
        # bond length from parent

        icoor_df = pandas.DataFrame(list(cattr.unstructure(restype.icoors))).set_index(
            "name"
        )
        names = [atom.name for atom in restype.atoms]

        par = torch.tensor(
            [restype.atom_to_idx[p] for p in icoor_df.loc[names]["parent"]],
            dtype=torch.long,
        )
        gpar = torch.tensor(
            [restype.atom_to_idx[gp] for gp in icoor_df.loc[names]["grand_parent"]],
            dtype=torch.long,
        )
        ggpar = torch.tensor(
            [
                (restype.atom_to_idx[ggp] if ggp in names else -1)
                for ggp in icoor_df.loc[names]["great_grand_parent"]
            ],
            dtype=torch.long,
        )

        atom_ancestors = torch.stack((par, gpar, ggpar), dim=0)

        def slide_neg1(t):
            t_rot = torch.full_like(t, -1)
            t_rot[t != -1] = res2rot[t != -1]
            return t_rot[rot2res]

        par_rot = slide_neg1(par)
        gpar_rot = slide_neg1(gpar)
        ggpar_rot = slide_neg1(ggpar)

        atom_ancestors_rot = torch.stack((par_rot, gpar_rot, ggpar_rot), dim=0)

        # chi spins the torsion for an atom if it is the last atom
        # atom for that chi or if its great-grand-parent is the last
        # atom for the chi and its parent is the second-to-last atom
        # for the chi. In this latter case, the a atom's ggp defines
        # an improper torsion.
        last_chi_ats = numpy.array(
            [restype.atom_to_idx[tor.d.atom] for tor in chi], dtype=int
        )
        second_to_last_chi_ats = numpy.array(
            [restype.atom_to_idx[tor.c.atom] for tor in chi], dtype=int
        )
        chi_that_spins_atom = torch.full((natoms,), -1, dtype=torch.long)
        chi_that_spins_atom[last_chi_ats] = torch.arange(
            len(last_chi_ats), dtype=torch.long
        )
        for i, at in enumerate(names):
            for j, (last, sectolast) in enumerate(
                zip(last_chi_ats, second_to_last_chi_ats)
            ):
                if sectolast == par[i] and chi_that_spins_atom[ggpar[i]] == j:
                    chi_that_spins_atom[i] = j

        atom_icoors = torch.tensor(
            [
                (row["phi"], row["theta"], row["d"])
                for _, row in icoor_df.loc[names].iterrows()
            ],
            dtype=torch.float64,
        )

        atom_icoors[chi_that_spins_atom != -1, 0] -= atom_icoors[
            last_chi_ats[chi_that_spins_atom[chi_that_spins_atom != -1]], 0
        ]
        atom_icoors = atom_icoors[rot2res]
        chi_that_spins_atom = chi_that_spins_atom[rot2res]

        sidechain_root = torch.full((3,), -1, dtype=torch.long)
        sc_root_name = restype.sidechain_building[sidechain].root
        root = restype.atom_to_idx[sc_root_name]
        sidechain_root[0] = res2rot[root]
        for at1, at2 in restype.bond_indices:
            if at1 == root and at2 in backbone_atom_inds:
                sidechain_root[1] = int(at2)
        for tor in chi:
            if tor.c.atom == sc_root_name:
                sidechain_root[2] = restype.atom_to_idx[tor.a.atom]
                break

        sidechain_root = res2rot[sidechain_root]

        sidechain_dfs = torch.cat(
            (
                res2rot[torch.tensor(nonsc_kept, dtype=torch.long)],
                res2rot[torch.tensor(dfs_ind, dtype=torch.long)],
            ),
            dim=0,
        )

        # sidechain_dfs = torch.full(
        #     (len(restype.sidechain_roots), natoms), -1, dtype=torch.long)

        # for i, sc in enumerate(restype.sidechain_roots):
        #     root = restype.atom_to_idx[sc]
        #     sidechain_roots[i, 0] = root
        #     for at1, at2 in restype.bonds:
        #         if at1 == sc and at2 in restype.backbone_atoms:
        #             bb_conn_at = restype.atom_to_idx[at2]
        #             sidechain_roots[i, 1] = bb_conn_at
        #             break
        #         elif at2 == sc and at1 in restype.backbone_atoms:
        #             bb_conn_at = restype.atom_to_idx[at1]
        #             sidechain_roots[i, 1] = bb_conn_at
        #     for tor in chi:
        #         if tor.c.atom == sc:
        #             sidechain_roots[i, 2] = restype.atom_to_idx[tor.a.atom]
        #             break
        #     assert sidechain_roots[i, 1] != -1
        #     assert sidechain_roots[i, 2] != -1
        #
        #     # temporarily erase the bond of the sc root
        #     bonds[root, bb_conn_at] = 0
        #     bonds[bb_conn_at, root] = 0
        #     dfs_ind, dfs_parent = scipy.sparse.csgraph.depth_first_order(bonds, root)
        #     sidechain_dfs[i, :len(dfs_ind)] = torch.tensor(dfs_ind)
        #     bonds[root, bb_conn_at] = 1
        #     bonds[bb_conn_at, root] = 1

        return cls(
            restype_name=restype.name,
            sidechain=sidechain,
            natoms=n_rotamer_atoms,
            rotatom_2_resatom=rot2res,
            resatom_2_rotatom=res2rot,
            bonds=bonds,
            is_backbone_atom=is_backbone_atom,
            backbone_atom_inds=backbone_atom_inds_rot,
            sidechain_root=sidechain_root,
            sidechain_dfs=sidechain_dfs,
            atom_icoors=atom_icoors,
            atom_ancestors=atom_ancestors_rot,
            chi_that_spins_atom=chi_that_spins_atom,
        )


@attr.s(auto_attribs=True, frozen=True)
class AASidechainBuilders:
    mapper: pandas.Index  # from restype name to the parameters that build it
    is_backbone_atom: Tensor(int)[:, :]
    backbone_atom_inds: Tensor(int)[:, :]
    sidechain_roots: Tensor(int)[:, :]
    sidechain_dfs: Tensor(int)[:, :, :]
    atom_icoors: Tensor(float)[:, :, 3]
    atom_ancestors: Tensor
