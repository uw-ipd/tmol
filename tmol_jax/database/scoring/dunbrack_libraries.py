"""NumPy-only scoring parameter records and required table loading."""

from __future__ import annotations

import os
from typing import Tuple

import attr
import numpy as np
import yaml

from ._structure import structure


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunMappingParams:
    dun_table_name: str
    residue_name: str


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericDataForAA:
    rotamers: np.ndarray
    rotamer_probabilities: np.ndarray
    rotamer_means: np.ndarray
    rotamer_stdvs: np.ndarray
    prob_sorted_rot_inds: np.ndarray
    backbone_dihedral_start: np.ndarray
    backbone_dihedral_step: np.ndarray
    rotamer_alias: np.ndarray
    backbone_is_mirrored: bool = False

    def nrotamers(self) -> int:
        return int(self.rotamers.shape[0])

    def nchi(self) -> int:
        return int(self.rotamers.shape[1])


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA
    non_rot_chi_start: float
    non_rot_chi_step: float
    non_rot_chi_period: float
    rotameric_chi_rotamers: np.ndarray
    nonrotameric_chi_probabilities: np.ndarray
    rotamer_boundaries: np.ndarray


@attr.s(auto_attribs=True, slots=True, frozen=True)
class _DunbrackLookupRaw:
    dunbrack_lookup: Tuple[DunMappingParams, ...]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackRotamerLibrary:
    dun_lookup: Tuple[DunMappingParams, ...]
    rotameric_libraries: Tuple[RotamericAADunbrackLibrary, ...] = ()
    semi_rotameric_libraries: Tuple[SemiRotamericAADunbrackLibrary, ...] = ()

    @staticmethod
    def load_libraries_from_npz(
        npz_path: str,
    ) -> Tuple[
        Tuple[RotamericAADunbrackLibrary, ...],
        Tuple[SemiRotamericAADunbrackLibrary, ...],
    ]:
        """Load the Dunbrack rotamer libraries from a numpy ``.npz``.

        The archive layout matches the one written by
        ``scripts/convert_binary_tables.py`` -- flat arrays keyed by
        ``rot{i}__...`` / ``semi{i}__...`` prefixes, plus ``__n_rot``
        and ``__n_semi`` counters.
        """
        rot_data_fields = (
            "rotamers",
            "rotamer_probabilities",
            "rotamer_means",
            "rotamer_stdvs",
            "prob_sorted_rot_inds",
            "backbone_dihedral_start",
            "backbone_dihedral_step",
            "rotamer_alias",
        )

        def _load_rot_data(data, prefix: str) -> RotamericDataForAA:
            mirror_key = prefix + "data__backbone_is_mirrored"
            return RotamericDataForAA(
                **{
                    field: np.asarray(data[prefix + "data__" + field]).copy()
                    for field in rot_data_fields
                },
                backbone_is_mirrored=(
                    bool(data[mirror_key]) if mirror_key in data else False
                ),
            )

        with np.load(npz_path, allow_pickle=False) as data:
            n_rot = int(data["__n_rot"])
            n_semi = int(data["__n_semi"])
            rot_libs_out = []
            for i in range(n_rot):
                prefix = f"rot{i}__"
                rot_libs_out.append(
                    RotamericAADunbrackLibrary(
                        table_name=str(data[prefix + "table_name"]),
                        rotameric_data=_load_rot_data(data, prefix),
                    )
                )
            semi_libs_out = []
            for i in range(n_semi):
                prefix = f"semi{i}__"
                semi_libs_out.append(
                    SemiRotamericAADunbrackLibrary(
                        table_name=str(data[prefix + "table_name"]),
                        rotameric_data=_load_rot_data(data, prefix),
                        non_rot_chi_start=float(data[prefix + "non_rot_chi_start"]),
                        non_rot_chi_step=float(data[prefix + "non_rot_chi_step"]),
                        non_rot_chi_period=float(data[prefix + "non_rot_chi_period"]),
                        rotameric_chi_rotamers=np.asarray(
                            data[prefix + "rotameric_chi_rotamers"]
                        ).copy(),
                        nonrotameric_chi_probabilities=np.asarray(
                            data[prefix + "nonrotameric_chi_probabilities"]
                        ).copy(),
                        rotamer_boundaries=np.asarray(
                            data[prefix + "rotamer_boundaries"]
                        ).copy(),
                    )
                )
        return (tuple(rot_libs_out), tuple(semi_libs_out))

    @classmethod
    def from_file(cls, path):
        with open(path) as stream:
            rows = structure(yaml.safe_load(stream), _DunbrackLookupRaw)
        stem = os.path.splitext(path)[0]
        rot, semi = cls.load_libraries_from_npz(stem + ".npz")
        return cls(rows.dunbrack_lookup, rot, semi)
