import torch
import attr
import cattr
import numpy
import pandas

import scipy.sparse.csgraph

from typing import Tuple

from tmol.types.attrs import ValidateAttrs
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.system.packed import PackedResidueSystem
from tmol.system.restypes import ResidueType
from tmol.database import ChemicalDatabase
from tmol.score.dunbrack.params import exclusive_cumsum


@attr.s(auto_attribs=True, frozen=True)
class SingleSidechainBuilder(ValidateAttrs):
    """Describe how to build the coordinates of the rotamers
    for a single sidechain on a single residue; each residue
    may have more than one sidechain
    """

    restype_name: str
    sidechain: int

    # how many atoms are built for this sidechain group?
    # it may be only a subset of the atoms for the residue;
    # it may also include the "down" or "up" atoms as virtuals
    natoms: int
    # the virtual connection atom indices, if present
    # position 0: the "down" atom
    # position 1: the "up" atom
    vconn_inds: Tensor(int)[2]

    # mapping of atoms built for this rotamer to the indices in the ResidueType
    # -1 for atoms that are not part of the residue
    rotatom_2_resatom: Tensor(int)[:]
    # mapping of the atoms of the original ResidueType to the indices in this rotamer
    # -1 for virtual connection atoms
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
    @validate_args
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

        # disconnect any cycles with the backbone that the residue type might have
        sc_building = restype.sidechain_building[sidechain]
        for aname, bname in sc_building.exclude_bonds:
            ai = restype.atom_to_idx[aname]
            bi = restype.atom_to_idx[bname]
            bonds[ai, bi] = 0
            bonds[bi, ai] = 0

        bbats = sc_building.backbone_atoms
        vconn_present = [(1 if "down" in bbats else 0), (1 if "up" in bbats else 0)]

        keep = numpy.zeros(natoms, dtype=int)
        nonsc = []

        names = [at.name for at in restype.atoms]
        for bbat in [restype.atom_to_idx[at] for at in bbats if at in names]:
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
        nonsc_set = set(nonsc)
        sc_root_at = restype.sidechain_building[sidechain].root
        if sc_root_at in bbats:
            # If there are no atoms in the sidechain, then a backbone atom
            # can be listed and then the only atoms for the "side chain"
            # that will be built are any hydrogen atoms for the backbone
            return keep, nonsc, vconn_present, torch.tensor([], dtype=torch.long)

        root = restype.atom_to_idx[sc_root_at]

        for other in range(natoms):
            if bonds[root, other] and keep[other]:
                bonds[root, other] = 0
                bonds[other, root] = 0
        bfs_ind, bfs_parent = scipy.sparse.csgraph.breadth_first_order(bonds, root)
        for ind in bfs_ind:
            # if the sidechain wraps around and is chemically bonded to
            # the backbone, then we have a problem; temporarily, what we'll
            # do is just not include the backbone atoms into the sidechain
            # build order; the more general form of this will be to
            # disconnect the sidechain from the backbone at a particular
            # bond before the BFS begins
            if ind not in nonsc_set:
                keep[ind] = 1
        return keep, nonsc, vconn_present, bfs_ind

    @classmethod
    @validate_args
    def from_restype(
        cls, chem_db: ChemicalDatabase, restype: ResidueType, sidechain: int
    ):

        # which atoms should be included?
        # It should be everything reachable by a DFS from the sidechain
        # root
        tmp = cls.determine_atoms_in_sidechain_group(chem_db, restype, sidechain)
        keep, nonsc_kept, vconn_present, dfs_ind = tmp

        names = [atom.name for atom in restype.atoms]
        natoms = len(restype.atoms)

        n_real_rotamer_atoms = sum(keep)
        n_rotamer_atoms = n_real_rotamer_atoms + sum(vconn_present)
        rot2res = torch.full((n_rotamer_atoms,), -1, dtype=torch.long)
        res2rot = torch.full((natoms,), -1, dtype=torch.long)
        count = 0
        for i, keep_i in enumerate(keep):
            if keep_i:
                res2rot[i] = count
                rot2res[count] = i
                count += 1

        rot2res_real = rot2res[rot2res != -1]

        vconn_inds = torch.full((2,), -1, dtype=torch.long)
        vcount = 0
        for i, present in enumerate(vconn_present):
            if present:
                vconn_inds[i] = int(n_real_rotamer_atoms + vcount)
                vcount += 1

        bonds = torch.zeros((natoms, natoms), dtype=torch.long)
        for ai, bi in restype.bond_indices:
            bonds[ai, bi] = 1
        bonds = bonds[rot2res_real, :]
        bonds = bonds[:, rot2res_real]
        rotamer_bonds = torch.zeros(
            (n_rotamer_atoms, n_rotamer_atoms), dtype=torch.long
        )
        rotamer_bonds[:n_real_rotamer_atoms, :n_real_rotamer_atoms] = bonds

        # sort the chi of the restypes to be ascending
        chi = sorted(
            [tor for tor in restype.torsions if "chi" in tor.name],
            key=lambda x: int(x.name.partition("chi")[2]),
        )
        nchi = len(chi)

        is_backbone_atom = torch.zeros((n_rotamer_atoms), dtype=torch.long)
        bbats = restype.sidechain_building[sidechain].backbone_atoms
        backbone_atom_inds = torch.tensor(
            [restype.atom_to_idx[at] for at in bbats if at in names], dtype=torch.long
        )
        backbone_atom_inds_rot = res2rot[backbone_atom_inds]
        is_backbone_atom[backbone_atom_inds] = 1
        for i in range(2):
            if vconn_inds[i] != -1:
                is_backbone_atom[vconn_inds] = 1

        ideal_dofs = torch.zeros((natoms, 3), dtype=torch.float)
        # dofs in order:
        # dihedral from great-grandparent
        # angle from grand parent
        # bond length from parent

        icoor_df = pandas.DataFrame(list(cattr.unstructure(restype.icoors))).set_index(
            "name"
        )

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
            t_rot[t != -1] = res2rot[t[t != -1]]
            t_rot = t_rot[rot2res_real]
            t_rot_return = torch.full((n_rotamer_atoms,), -1, dtype=torch.long)
            t_rot_return[:n_real_rotamer_atoms] = t_rot
            return t_rot_return

        par_rot = slide_neg1(par)
        gpar_rot = slide_neg1(gpar)
        ggpar_rot = slide_neg1(ggpar)

        atom_ancestors_rot = torch.stack((par_rot, gpar_rot, ggpar_rot), dim=1)

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
        atom_icoors = atom_icoors[rot2res_real]
        atom_icoors_rotamer = torch.zeros((n_rotamer_atoms, 3), dtype=torch.float)
        atom_icoors_rotamer[:n_real_rotamer_atoms, :] = atom_icoors
        chi_that_spins_atom = chi_that_spins_atom[rot2res_real]
        tmp = chi_that_spins_atom
        chi_that_spins_atom = torch.full((n_rotamer_atoms,), -1, dtype=torch.long)
        chi_that_spins_atom[:n_real_rotamer_atoms] = tmp

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
            (res2rot[nonsc_kept], vconn_inds[vconn_inds != -1], res2rot[dfs_ind]), dim=0
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
            natoms=int(n_rotamer_atoms),
            vconn_inds=vconn_inds,
            rotatom_2_resatom=rot2res,
            resatom_2_rotatom=res2rot,
            bonds=rotamer_bonds,
            is_backbone_atom=is_backbone_atom,
            backbone_atom_inds=backbone_atom_inds_rot,
            sidechain_root=sidechain_root,
            sidechain_dfs=sidechain_dfs,
            atom_icoors=atom_icoors_rotamer,
            atom_ancestors=atom_ancestors_rot,
            chi_that_spins_atom=chi_that_spins_atom,
        )


@attr.s(auto_attribs=True, frozen=True)
class SidechainBuilders:
    mapper: pandas.Index  # from restype name to the restype index
    n_sc_for_restype: Tensor(torch.long)[:]  # the number of scs for each restype
    restype_offsets: Tensor(torch.long)[:]  # the sc param offsets for a restype
    n_atoms: Tensor(torch.long)[:]  # the number of atoms for an sc
    vconn_inds: Tensor(torch.long)[:, 2]
    rotatom_2_resatom: Tensor(torch.long)[:, :]
    resatom_2_rotatom: Tensor(torch.long)[:, :]
    bonds: Tensor(torch.long)[:, :, :]
    is_backbone_atom: Tensor(torch.long)[:, :]  # is a particular atom a backbone atom
    n_backbone_atoms: Tensor(torch.long)[:]
    backbone_atom_inds: Tensor(torch.long)[:, :]
    sidechain_roots: Tensor(torch.long)[:, 3]
    sidechain_dfs: Tensor(torch.long)[:, :]
    atom_icoors: Tensor(float)[:, :, 3]
    atom_ancestors: Tensor(torch.long)[:, :]
    chi_that_spins_atom: Tensor(torch.long)[:, :]

    @classmethod
    def from_restypes(
        cls, chem_db: ChemicalDatabase, restypes: Tuple[ResidueType, ...]
    ):
        """Construct the coalesced set of sidechain construction data
        from an input set of ResidueType objects"""

        n_restypes = len(restypes)
        builders = [
            SingleSidechainBuilder.from_restype(chem_db, rt, sc)
            for rt in restypes
            for sc in range(len(rt.sidechain_building))
        ]
        nbuilders = len(builders)
        mapper = pandas.Index([rt.name for rt in restypes])

        n_sc_for_restype = cls.n_sc_from_restypes(restypes)
        restype_offsets = exclusive_cumsum(n_sc_for_restype)
        n_atoms = cls.natoms_from_builders(builders)

        max_rotamer_natoms = torch.max(n_atoms)
        max_restype_natoms = max(
            builder.resatom_2_rotatom.shape[0] for builder in builders
        )

        vconn_inds = cls.vconn_inds_from_builders(builders)

        rotatom_2_resatom = cls.rot2res_from_builders(builders, max_rotamer_natoms)
        resatom_2_rotatom = cls.res2rot_from_builders(builders, max_restype_natoms)
        bonds = cls.bonds_from_builders(builders, max_rotamer_natoms)
        is_backbone_atom = cls.is_backbone_from_builders(builders, max_rotamer_natoms)
        n_backbone_atoms = cls.n_bbats_from_builders(builders)

        backbone_atom_inds = cls.bbats_from_builders(builders)
        sidechain_roots = cls.sidechain_roots_from_builders(builders)
        sidechain_dfs = cls.sidechain_dfs_from_builders(builders, max_rotamer_natoms)
        atom_icoors = cls.atom_icoors_from_builders(builders, max_rotamer_natoms)
        atom_ancestors = cls.atom_ancestors_from_builders(builders, max_rotamer_natoms)
        chi_that_spins_atom = cls.chi_spinner_from_builders(
            builders, max_rotamer_natoms
        )

        return cls(
            mapper=mapper,
            n_sc_for_restype=n_sc_for_restype,
            restype_offsets=restype_offsets,
            n_atoms=n_atoms,
            vconn_inds=vconn_inds,
            rotatom_2_resatom=rotatom_2_resatom,
            resatom_2_rotatom=resatom_2_rotatom,
            bonds=bonds,
            is_backbone_atom=is_backbone_atom,
            n_backbone_atoms=n_backbone_atoms,
            backbone_atom_inds=backbone_atom_inds,
            sidechain_roots=sidechain_roots,
            sidechain_dfs=sidechain_dfs,
            atom_icoors=atom_icoors,
            atom_ancestors=atom_ancestors,
            chi_that_spins_atom=chi_that_spins_atom,
        )

    @classmethod
    def n_sc_from_restypes(cls, restypes):
        return torch.tensor(
            [len(rt.sidechain_building) for rt in restypes], dtype=torch.long
        )

    @classmethod
    def natoms_from_builders(cls, builders):
        return torch.tensor([builder.natoms for builder in builders], dtype=torch.long)

    @classmethod
    def vconn_inds_from_builders(cls, builders):
        return torch.stack(tuple(builder.vconn_inds for builder in builders))

    @classmethod
    def rot2res_from_builders(cls, builders, max_rotamer_natoms):
        rotatom_2_resatom = torch.full(
            (len(builders), max_rotamer_natoms), -1, dtype=torch.long
        )
        for i, builder in enumerate(builders):
            nrotatoms = builder.rotatom_2_resatom.shape[0]
            rotatom_2_resatom[i, :nrotatoms] = builder.rotatom_2_resatom
        return rotatom_2_resatom

    @classmethod
    def res2rot_from_builders(cls, builders, max_restype_natoms):
        resatom_2_rotatom = torch.full(
            (len(builders), max_restype_natoms), -1, dtype=torch.long
        )
        for i, builder in enumerate(builders):
            nresatoms = builder.resatom_2_rotatom.shape[0]
            resatom_2_rotatom[i, :nresatoms] = builder.resatom_2_rotatom
        return resatom_2_rotatom

    @classmethod
    def bonds_from_builders(cls, builders, max_rotamer_natoms):
        bonds = torch.zeros(
            (len(builders), max_rotamer_natoms, max_rotamer_natoms), dtype=torch.long
        )
        for i, builder in enumerate(builders):
            natoms = builder.natoms
            bonds[i, :natoms, :natoms] = builder.bonds
        return bonds

    @classmethod
    def is_backbone_from_builders(cls, builders, max_rotamer_natoms):
        is_backbone_atom = torch.zeros(
            (len(builders), max_rotamer_natoms), dtype=torch.long
        )
        for i, builder in enumerate(builders):
            natoms = builder.natoms
            is_backbone_atom[i, :natoms] = builder.is_backbone_atom
        return is_backbone_atom

    @classmethod
    def n_bbats_from_builders(cls, builders):
        return torch.tensor(
            [len(b.backbone_atom_inds) for b in builders], dtype=torch.long
        )

    @classmethod
    def bbats_from_builders(cls, builders):
        max_n_backbone_atoms = max(
            builder.backbone_atom_inds.shape[0] for builder in builders
        )
        backbone_atom_inds = torch.zeros(
            (len(builders), max_n_backbone_atoms), dtype=torch.long
        )
        for i, builder in enumerate(builders):
            nbb = builder.backbone_atom_inds.shape[0]
            backbone_atom_inds[i, :nbb] = builder.backbone_atom_inds
        return backbone_atom_inds

    @classmethod
    def sidechain_roots_from_builders(cls, builders):
        return torch.stack(tuple(builder.sidechain_root for builder in builders))

    @classmethod
    def sidechain_dfs_from_builders(cls, builders, max_rotamer_natoms):
        sidechain_dfs = torch.full(
            (len(builders), max_rotamer_natoms), -1, dtype=torch.long
        )
        for i, builder in enumerate(builders):
            natoms = builder.natoms
            sidechain_dfs[i, :natoms] = builder.sidechain_dfs
        return sidechain_dfs

    @classmethod
    def atom_icoors_from_builders(cls, builders, max_rotamer_natoms):
        atom_icoors = torch.zeros(
            (len(builders), max_rotamer_natoms, 3), dtype=torch.float
        )
        for i, builder in enumerate(builders):
            natoms = builder.natoms
            atom_icoors[i, :natoms, :] = builder.atom_icoors
        return atom_icoors

    @classmethod
    def atom_ancestors_from_builders(cls, builders, max_rotamer_natoms):
        atom_ancestors = torch.zeros(
            (len(builders), max_rotamer_natoms, 3), dtype=torch.float
        )
        for i, builder in enumerate(builders):
            natoms = builder.natoms
            atom_ancestors[i, :natoms] = builder.atom_ancestors
        return atom_ancestors

    @classmethod
    def chi_spinner_from_builders(cls, builders, max_rotamer_natoms):
        chi_that_spins_atom = torch.zeros(
            (len(builders), max_rotamer_natoms), dtype=torch.long
        )
        for i, builder in enumerate(builders):
            natoms = builder.natoms
            chi_that_spins_atom[i, :natoms] = builder.chi_that_spins_atom
        return chi_that_spins_atom
