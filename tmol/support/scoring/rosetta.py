import toolz
import pickle

import attr

import numpy
import pandas


def strip_variants(pose):
    import pyrosetta.distributed.packed_pose as packed_pose

    pose = packed_pose.to_pose(pose)

    import pyrosetta.rosetta.core as core

    for i, r in enumerate(pose.residues):
        for t in r.type().variant_type_enums():
            core.pose.remove_variant_type_from_pose_residue(pose, t, i + 1)

    return pose


@attr.s(slots=True)
class PoseScoreWrapper:

    pickled_pose: str = attr.ib()
    rosetta_version: str = attr.ib()
    atoms: pandas.DataFrame = attr.ib()
    residue_scores: numpy.ndarray = attr.ib()
    total_scores: numpy.ndarray = attr.ib()
    hbonds: pandas.DataFrame = attr.ib()

    @classmethod
    def from_pose(cls, pose):
        from tmol.support.rosetta.init import pyrosetta

        pyrosetta.get_score_function()(pose)

        atoms = pandas.DataFrame.from_records(
            [
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
            ]
        )

        residue_scores = pose.energies().residue_total_energies_array()
        total_scores = pose.energies().total_energies_array()

        hbset = pyrosetta.rosetta.core.scoring.hbonds.HBondSet()
        pyrosetta.rosetta.core.scoring.hbonds.fill_hbond_set(pose, False, hbset)

        hbonds = pandas.DataFrame.from_records(
            [
                {
                    "a_res": hbond.acc_res() - 1,
                    "a_atom": pose.residue(hbond.acc_res())
                    .atom_name(hbond.acc_atm())
                    .strip(),
                    "h_res": hbond.don_res() - 1,
                    "h_atom": pose.residue(hbond.don_res())
                    .atom_name(hbond.don_hatm())
                    .strip(),
                    "energy": hbond.energy(),
                }
                for hbond in (hbset.hbond(i + 1) for i in range(hbset.nhbonds()))
            ]
        )

        import pyrosetta

        pyrosetta.distributed.packed_pose.PackedPose

        return cls(
            rosetta_version=pyrosetta.rosetta.utility.Version.version(),
            pickled_pose=pickle.dumps(pose),
            atoms=atoms,
            residue_scores=residue_scores,
            total_scores=total_scores,
            hbonds=hbonds,
        )

    @classmethod
    def from_pdbstring(cls, pdbstring):
        from pyrosetta.distributed.io import pose_from_pdbstring

        return cls.from_pose(
            toolz.compose(strip_variants, pose_from_pdbstring)(pdbstring)
        )

    @property
    def tmol_residues(self):
        from tmol.system.io import ResidueReader

        return [
            ResidueReader.get_default().parse_atom_block(atoms)
            for (chaini, resi), atoms in self.atoms.groupby(["chaini", "resi"])
        ]

    @property
    def pose(self):
        import tmol.support.rosetta.init  # noqa Import to ensure init.

        return pickle.loads(self.pickled_pose)
