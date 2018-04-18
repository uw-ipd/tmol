import pyrosetta.rosetta as rosetta

import pyrosetta.distributed.io as io
import pyrosetta.distributed.tasks.score as score
import pyrosetta.distributed.packed_pose as packed_pose

import toolz

import tmol.support.rosetta.init  # noqa

import attr
import pandas


def strip_variants(pose):
    pose = packed_pose.to_pose(pose)

    import pyrosetta.rosetta.core as core
    for i, r in enumerate(pose.residues):
        for t in r.type().variant_type_enums():
            core.pose.remove_variant_type_from_pose_residue(pose, t, i + 1)

    return packed_pose.to_packed(pose)


@attr.s(slots=True)
class PoseScoreWrapper:
    @classmethod
    def from_pdbstring(cls, pdbstring):
        return toolz.compose(
            cls,
            score.ScorePoseTask(),
            strip_variants,
            io.pose_from_pdbstring,
        )(pdbstring)

    pose = attr.ib(converter=packed_pose.to_packed)

    atoms = attr.ib()

    @atoms.default
    def _load_atoms(self):
        pose = self.pose.pose

        return pandas.DataFrame.from_records([
            dict(
                chaini=pose.chain(ri + 1),
                resi=ri + 1,
                atomi=ai + 1,
                resn=r.name3(),
                atomn=r.atom_name(ai + 1).strip(),
                x=r.xyz(ai + 1)[0],
                y=r.xyz(ai + 1)[1],
                z=r.xyz(ai + 1)[2],
            )
            for ri, r in enumerate(pose.residues)
            for ai in range(r.natoms())
        ])

    residue_scores = attr.ib()

    @residue_scores.default
    def _load_residue_scores(self):
        return self.pose.pose.energies().residue_total_energies_array()

    total_scores = attr.ib()

    @total_scores.default
    def _load_total_scores(self):
        return self.pose.pose.energies().total_energies_array()

    hbonds = attr.ib()

    @hbonds.default
    def _load_hbonds(self):
        pose = self.pose.pose

        hbset = rosetta.core.scoring.hbonds.HBondSet()
        rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbset)

        return pandas.DataFrame.from_records([{
            "a_res": hbond.acc_res() - 1,
            "a_atom":
                pose.residue(hbond.acc_res()).atom_name(hbond.acc_atm())
                .strip(),
            "h_res": hbond.don_res() - 1,
            "h_atom":
                pose.residue(hbond.don_res()).atom_name(hbond.don_hatm())
                .strip(),
            "energy": hbond.energy(),
        } for hbond in (hbset.hbond(i + 1) for i in range(hbset.nhbonds()))])

    @property
    def tmol_residues(self):
        from tmol.system.residue.io import ResidueReader
        reader = ResidueReader()

        return [
            reader.parse_atom_block(atoms)
            for (chaini, resi), atoms in self.atoms.groupby(["chaini", "resi"])
        ]
