import attr
import cattr
import numpy
import yaml
import zarr

from typing import Tuple

from tmol.types.array import NDArray


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepMappingParams:
    table_id: str
    res_middle: str
    res_upper: str = "_"
    invert_phi: bool = False
    invert_psi: bool = False


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepTables:
    table_id: str
    mu: NDArray[float]
    sigma: NDArray[float]
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


def load_bbdep_omega_tables_from_zarr(path_tables):
    alltables = []
    store = zarr.ZipStore(path_tables, mode="r")
    root = zarr.Group(store)
    for aa in root:
        alltables.append(
            OmegaBBDepTables(
                table_id=aa,
                mu=numpy.array(root[aa]["mu"]),
                sigma=numpy.array(root[aa]["sigma"]),
                bbstep=root[aa]["mu"].attrs["bbstep"],
                bbstart=root[aa]["mu"].attrs["bbstart"],
            )
        )
    return alltables


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepDatabase:
    uniq_id: str  # unique id for memoization
    bbdep_omega_lookup: Tuple[OmegaBBDepMappingParams, ...]
    bbdep_omega_tables: Tuple[OmegaBBDepTables, ...]

    @classmethod
    def from_files(cls, path_lookup, path_tables):
        with open(path_lookup, "r") as infile_lookup:
            raw = yaml.safe_load(infile_lookup)
            bbdep_omega_lookup = cattr.structure(
                raw["omega_bbdep_lookup"], attr.fields(cls).bbdep_omega_lookup.type
            )

        bbdep_omega_tables = load_bbdep_omega_tables_from_zarr(path_tables)

        uniq_id = path_lookup + "," + path_tables

        return cls(
            uniq_id=uniq_id,
            bbdep_omega_lookup=bbdep_omega_lookup,
            bbdep_omega_tables=bbdep_omega_tables,
        )
