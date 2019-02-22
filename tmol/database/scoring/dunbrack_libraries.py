import attr
import cattr
import json
import zarr
import torch
from typing import Tuple

from tmol.types.torch import Tensor


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericDataForAA:
    rotamers: Tensor(int)[:, :]  # nrotamers x nchi
    rotamer_probabilities: Tensor(float)  # (1 + n-bb) dimensional table
    rotamer_means: Tensor(
        float
    )  # (1 + n-bb + 1) dimensional table: nrots x [nbb] x nchi
    rotamer_stdvs: Tensor(
        float
    )  # ( 1 + n-bb + 1 ) dimensional table: nrotx x [nbb] x nchi
    prob_sorted_rot_inds: Tensor(int)  # (n-bb+1)-dimensional table
    backbone_dihedral_start: Tensor(float)[:]
    backbone_dihedral_step: Tensor(float)[:]

    @classmethod
    def from_zgroup(cls, zgroup):
        rotgrp = zgroup["rotameric_data"]
        rotamers = torch.tensor(rotgrp["rotamers"][...], dtype=torch.long)
        rot_probs = torch.tensor(rotgrp["probabilities"][...], dtype=torch.float)
        rot_means = torch.tensor(rotgrp["means"][...], dtype=torch.float)
        rot_stdvs = torch.tensor(rotgrp["stdvs"][...], dtype=torch.float)
        prob_sorted_rot_inds = torch.tensor(
            rotgrp["prob_sorted_rot_inds"][...], dtype=torch.long
        )
        bb_dihe_start = torch.tensor(
            rotgrp["backbone_dihedral_start"][...], dtype=torch.float
        )
        bb_dihe_step = torch.tensor(
            rotgrp["backbone_dihedral_step"][...], dtype=torch.float
        )

        assert len(rotamers.shape) == 2
        assert bb_dihe_start.shape[0] == bb_dihe_step.shape[0]
        assert len(rot_probs.shape) == bb_dihe_start.shape[0] + 1
        assert len(rot_means.shape) == bb_dihe_start.shape[0] + 2
        assert len(rot_stdvs.shape) == bb_dihe_start.shape[0] + 2
        assert len(prob_sorted_rot_inds.shape) == bb_dihe_start.shape[0] + 1

        assert rotamers.shape[0] == rot_probs.shape[0]
        assert rotamers.shape[0] == rot_means.shape[0]
        assert rotamers.shape[0] == rot_stdvs.shape[0]
        assert rotamers.shape[1] == rot_means.shape[-1]
        assert rotamers.shape[1] == rot_stdvs.shape[-1]

        nbb_samps = torch.div(torch.ones((1,), dtype=torch.float) * 360, bb_dihe_step)
        for bb in range(len(bb_dihe_start)):
            assert rot_probs.shape[1 + bb] == nbb_samps[bb]
            assert rot_means.shape[1 + bb] == nbb_samps[bb]
            assert rot_stdvs.shape[1 + bb] == nbb_samps[bb]

        return cls(
            rotamers=rotamers,
            rotamer_probabilities=rot_probs,
            rotamer_means=rot_means,
            rotamer_stdvs=rot_stdvs,
            prob_sorted_rot_inds=prob_sorted_rot_inds,
            backbone_dihedral_start=bb_dihe_start,
            backbone_dihedral_step=bb_dihe_step,
        )

    def nrotamers(self):
        return self.rotamers.shape[0]

    def nchi(self):
        return self.rotamers.shape[1]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA

    @classmethod
    def from_zgroup(cls, zgroup, name):
        lib_group = zgroup[name]
        rotameric_data = RotamericDataForAA.from_zgroup(lib_group)
        return cls(table_name=name, rotameric_data=rotameric_data)


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA
    non_rot_chi_start: float
    non_rot_chi_step: float
    non_rot_chi_period: float  # 180 or 360, e.g.
    rotameric_chi_rotamers: Tensor(int)[:, :]  # nrots x n-rotameric-chi
    nonrotameric_chi_probabilities: Tensor(float)  # (1+nbb+1)-dimensional table
    rotamer_boundaries: Tensor(float)[:, 2]  # 2nd dimension: 0=left, 1=right

    @classmethod
    def from_zgroup(cls, zgroup, name):
        semirot_group = zgroup[name]
        rotameric_data = RotamericDataForAA.from_zgroup(semirot_group)
        non_rot_chi_start = semirot_group.attrs["nonrot_chi_start"]
        non_rot_chi_step = semirot_group.attrs["nonrot_chi_step"]
        non_rot_chi_period = semirot_group.attrs["nonrot_chi_period"]
        rotameric_chi_rotamers = torch.tensor(
            semirot_group["rotameric_chi_rotamers"][...], dtype=torch.long
        )
        nonrotameric_chi_probabilities = torch.tensor(
            semirot_group["nonrotameric_chi_probabilities"][...], dtype=torch.float
        )
        rotamer_boundaries = torch.tensor(
            semirot_group["rotamer_boundaries"][...], dtype=torch.long
        )

        rot_probs = rotameric_data.rotamer_probabilities
        # print("rot_probs.shape",rot_probs.shape)
        # print("nonrotameric_chi_probabilities.shape",nonrotameric_chi_probabilities.shape)
        for i in range(1, len(rot_probs.shape)):
            assert rot_probs.shape[i] == nonrotameric_chi_probabilities.shape[i]
        assert (
            nonrotameric_chi_probabilities.shape[0] == rotameric_chi_rotamers.shape[0]
        )
        assert rotameric_data.nchi() == rotameric_chi_rotamers.shape[1] + 1
        assert (
            non_rot_chi_period // non_rot_chi_step
            == nonrotameric_chi_probabilities.shape[-1]
        )
        # print("rotamer_boundaries.shape[0] == rotameric_data.nrotamers()", rotamer_boundaries.shape[0],"==", rotameric_data.nrotamers())
        assert rotamer_boundaries.shape[0] == rotameric_data.nrotamers()
        assert rotamer_boundaries.shape[1] == 2

        return cls(
            table_name=name,
            rotameric_data=rotameric_data,
            non_rot_chi_start=non_rot_chi_start,
            non_rot_chi_step=non_rot_chi_step,
            non_rot_chi_period=non_rot_chi_period,
            rotameric_chi_rotamers=rotameric_chi_rotamers,
            nonrotameric_chi_probabilities=nonrotameric_chi_probabilities,
            rotamer_boundaries=rotamer_boundaries,
        )


@attrs.s(auto_attribs=True, slots=True, frozen=True)
class DunMappingParams:
    dun_table_name: str
    residue_name: str
    invert_bb: Tuple[bool, ...]


def load_tables_from_zarr(path_tables):
    store = zarr.ZipStore(path_tables)
    zgroup = zarr.group(store=store)
    rotameric_group = zgroup["rotameric_tables"]
    table_name_list = rotameric_group.attrs["tables"]
    rotameric_libraries = []
    for table in table_name_list:
        rotameric_libraries.append(
            RotamericAADunbrackLibrary.from_zgroup(rotameric_group, table)
        )

    semirotameric_group = zgroup["semirotameric_tables"]
    table_name_list = semirotameric_group.attrs["tables"]
    semi_rotameric_libraries = []
    for table in table_name_list:
        semi_rotameric_libraries.append(
            SemiRotamericAADunbrackLibrary.from_zgroup(semirotameric_group, table)
        )
    return rotameric_libraries, semi_rotameric_libraries


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackRotamerLibrary:
    dun_lookup: Tuple[DunMappingParams, ...]
    rotameric_libraries: Tuple[RotamericAADunbrackLibrary, ...]
    semi_rotameric_libraries: Tuple[SemiRotamericAADunbrackLibrary, ...]

    @classmethod
    def from_zarr_archive(cls, path_lookup, path_tables):

        with open(path_lookup, "r") as infile_lookup:
            raw = yaml.load(infile_lookup)
            dun_lookup = cattr.structure(
                raw["dun_lookup"], attr.fields(cls).dun_lookup.type
            )

        rotameric_libraries, semi_rotameric_libraries = load_tables_from_zarr(
            path_tables
        )

        return DunbrackRotamerLibrary(
            dun_lookup=dun_lookup,
            rotameric_libraries=rotameric_libraries,
            semi_rotameric_libraries=semi_rotameric_libraries,
        )


@attr.s(auto_attribs=True, slots=True, frozen=True)
class CompactedDunbrackRotamerLibrary:

    bbstart: Tensor(torch.float)[:, :]  # ntables x nbb
    bbstep: Tensor(torch.float)[:, :]  # ntables x nbb

    offset_for_prob_rotameric_table: Tuple[int, ...]
    rotameric_prob_interp_tables: Tuple[Tensor(torch.float), ...]  # nbb-dim tensors

    offset_for_mean_rotameric_table: Tuple[int, ...]
    rotameric_mean_interp_tables: Tuple[Tensor(torch.float), ...]  # nbb-dim tensors
    rotameric_sdev_interp_tables: Tuple[Tensor(torch.float), ...]  # nbb-dim tensors

    non_rot_chi_start: Tensor(torch.float)[:]
    non_rot_chi_step: Tensor(torch.float)[:]
    non_rot_chi_period: Tensor(torch.float)[:]
    offset_for_semirotameric_table: Tuple[int, ...]
    semirotameric_prob_interp_tables: Tuple[
        Tensor(torch.float), ...
    ]  # nbb+1-dim tensors

    @classmethod
    def from_dunbrack_rotlib(cls, drl):
        # here we will create an indexing of the rotameric tables
        # including the semi-rotameric tables
        # and for each one will create the interpolation coefficients
        # tensor
        #
        pass
