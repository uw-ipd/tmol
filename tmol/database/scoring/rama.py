import attr
import cattr
import json
import toolz.functoolz
import zarr
import numpy
import os
import shutil

from typing import Tuple

import torch

from frozendict import frozendict
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from tmol.numeric.bspline import BSplineInterpolation


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaTable:
    name: str
    phi_step: float
    psi_step: float
    phi_start: float
    psi_start: float
    probabilities: Tensor(torch.float)[:, :]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class EvaluationMapping:
    condition: str
    table_name: str


@attr.s(auto_attribs=True, frozen=True, slots=True, hash=False, repr=False)
class RamaDBFromText:
    tables: Tuple[RamaTable, ...]
    evaluation_mappings: Tuple[EvaluationMapping, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaSingleMapper:
    central_res: str
    upper_res: str
    upper_res_pos: bool
    cond_ndots: int
    which_table: int

    @staticmethod
    @validate_args
    def from_condition(condition: str, which_table: int):
        """Turn the condition string in rama.json into the parts that
        decide which table to use for each residue"""
        cond_parts = condition.partition(",")
        cent_res = cond_parts[0]
        upper_res_parts = cond_parts[2][1:-1].partition(":")
        upper_res2 = upper_res_parts[2]
        upper_res_pos = upper_res2[0] != "!"
        upper_res = upper_res_parts[2] if upper_res_pos else upper_res_parts[2][1:]
        cond_ndots = len(cent_res.split(".")) - 1

        return RamaSingleMapper(
            central_res=cent_res,
            upper_res=upper_res,
            upper_res_pos=upper_res_pos,
            cond_ndots=cond_ndots,
            which_table=which_table,
        )

    def matches(
        self, cent_res_props: Tuple[str, ...], upper_res_props: Tuple[str, ...]
    ) -> bool:
        cent_matches = False
        for prop in cent_res_props:
            if prop.startswith(self.central_res):
                cent_matches = True
                break
        if not cent_matches:
            return False

        upper_matches = False
        for prop in upper_res_props:
            if prop.startswith(self.upper_res):
                upper_matches = True
                break
        return upper_matches == self.upper_res_pos


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaMapper:
    """Map properties for residue i and i+1 into a table index

    The amino acid properties for residues i and i+1 will be examined
    to answer the question: "which of the various tables that have been
    defined in the input file should be used for residue i?"
    """

    mappers: Tuple[RamaSingleMapper, ...]
    mappers_by_centres: frozendict
    ndots_to_consider: Tuple[int]

    @validate_args
    def from_eval_mapping_and_table_list(
        evaluation_mappings: Tuple[EvaluationMapping, ...],
        tables: Tuple[RamaTable, ...],
    ):
        ndots_to_consider = set([])
        mappers = []
        mappers_by_centres = {}
        for ev_map in evaluation_mappings:
            try:
                tab_ind = next(
                    i for i, e in enumerate(tables) if e.name == ev_map.table_name
                )
            except StopIteration:
                raise ValueError(
                    'Ramachandran table requested for mapping "'
                    + ev_map.condition
                    + '" -> '
                    + ev_map.table_name
                    + " does not exist."
                )
            map1 = RamaSingleMapper.from_condition(ev_map.condition, tab_ind)
            ndots_to_consider.add(map1.cond_ndots)
            mappers.append(map1)
            if map1.central_res not in mappers_by_centres:
                mappers_by_centres[map1.central_res] = [map1]
            else:
                mappers_by_centres[map1.central_res].append(map1)
        mappers_by_centres = frozendict(mappers_by_centres)
        ndots_to_consider = tuple(ndots_to_consider)
        return RamaMapper(
            mappers=mappers,
            mappers_by_centres=mappers_by_centres,
            ndots_to_consider=ndots_to_consider,
        )

    def substr_end_for_ndotted_prefix(self, property: str, prefix_ndots: int) -> int:
        """Return the index of the prefix_ndots+1'th dot in the given property string"""
        count_dots = 0
        for i in range(len(property)):
            if property[i] == ".":
                count_dots += 1
                if count_dots > prefix_ndots:
                    return i
        return len(property)

    def table_ind_for_res(
        self, cent_res_props: Tuple[str, ...], upper_res_props: Tuple[str, ...]
    ) -> int:
        # look for mappers that use the cent_res
        # and then check each mapper using it
        for prop in cent_res_props:
            for prefix_ndots in self.ndots_to_consider:
                last = self.substr_end_for_ndotted_prefix(prop, prefix_ndots)
                prefix = prop[:last]
                if prefix not in self.mappers_by_centres:
                    continue
                mappers = self.mappers_by_centres[prefix]
                for mapper in mappers:
                    if mapper.matches(cent_res_props, upper_res_props):
                        return mapper.which_table
        return -1


@attr.s(auto_attribs=True, frozen=True, slots=True, hash=False, repr=False)
class RamaDatabase:
    sourcefilepath: str
    tables: Tuple[RamaTable, ...]
    evaluation_mappings: Tuple[EvaluationMapping, ...]
    mapper: RamaMapper

    def __hash__(self):
        return hash(self.sourcefilepath)

    def __repr__(self):
        return "RamaDatabase(%s)" % self.sourcefilepath

    @classmethod
    def from_file(cls, path, read_binary=True, write_binary=True):
        """Load Ramachandran tables from JSON.

        All tables in the json file must have the same phi & psi step sizes.
        If there is a zarr binary representation of the data already
        present, then load from that instead of the JSON file. If the
        zarr binary representation of the data is not yet present, then
        save the binary format.
        """
        converter = cattr.Converter()
        converter.register_structure_hook(
            Tensor(torch.float)[:, :], lambda arr, _: torch.FloatTensor(arr)
        )

        bin_fname = cls.binary_filename_for_path(path)
        if os.path.isdir(bin_fname) and read_binary:
            prelim_rep = cls.load_textrep_from_binary(path)
        else:
            with open(path, "r") as infile:
                raw = json.load(infile)
            prelim_rep = converter.structure(raw, RamaDBFromText)
            for i, tab in enumerate(prelim_rep.tables):
                if i == 0:
                    phi_step = tab.phi_step
                    psi_step = tab.psi_step
                assert phi_step == tab.phi_step
                assert psi_step == tab.psi_step

        instance = cls.from_ramadb_from_text(path, prelim_rep)
        if not os.path.isdir(bin_fname) and write_binary:
            instance.save_to_binary()
        return instance

    @classmethod
    def from_ramadb_from_text(cls, path, prelim_rep):
        mapper = RamaMapper.from_eval_mapping_and_table_list(
            prelim_rep.evaluation_mappings, prelim_rep.tables
        )
        return cls(
            sourcefilepath=path,
            tables=prelim_rep.tables,
            evaluation_mappings=prelim_rep.evaluation_mappings,
            mapper=mapper,
        )

    @classmethod
    def binary_filename_for_path(cls, path):
        return path + ".bin"

    @classmethod
    def clear_binary_rep_for_file(cls, path):
        bin_fname = cls.binary_filename_for_path(path)
        if os.path.isdir(bin_fname):
            shutil.rmtree(bin_fname)

    def save_to_binary(self):
        # save the tables to binary

        store = zarr.storage.LMDBStore(
            self.binary_filename_for_path(self.sourcefilepath)
        )
        zarr_group = zarr.group(store=store)
        table_names = []
        for table in self.tables:
            table_names.append(table.name)
            tgroup = zarr_group.create_group(table.name)
            tgroup.attrs["phi_step"] = table.phi_step
            tgroup.attrs["psi_step"] = table.psi_step
            tgroup.attrs["phi_start"] = table.phi_start
            tgroup.attrs["psi_start"] = table.psi_start
            tgroup.array("probabilities", numpy.array(table.probabilities))
            # tgroup.array("energies", numpy.array(table.energies))
        zarr_group.attrs["tables"] = table_names
        zarr_group.attrs["eval_map"] = [
            (x.condition, x.table_name) for x in self.evaluation_mappings
        ]
        store.close()

    @classmethod
    def load_textrep_from_binary(cls, path):
        store = zarr.storage.LMDBStore(cls.binary_filename_for_path(path))
        zarr_group = zarr.group(store=store)
        tables = []
        evaluation_mappings = []
        table_names = zarr_group.attrs["tables"]
        for table_name in table_names:
            tgroup = zarr_group[table_name]
            phi_step = tgroup.attrs["phi_step"]
            psi_step = tgroup.attrs["psi_step"]
            phi_start = tgroup.attrs["phi_start"]
            psi_start = tgroup.attrs["psi_start"]
            prob_table = tgroup["probabilities"][:]
            prob_table = torch.FloatTensor(prob_table[:])
            tables.append(
                RamaTable(
                    name=table_name,
                    phi_step=phi_step,
                    psi_step=psi_step,
                    phi_start=phi_start,
                    psi_start=psi_start,
                    probabilities=prob_table,
                )
            )
        tables = tuple(tables)
        mappings = zarr_group.attrs["eval_map"]
        eval_mappings = tuple(
            EvaluationMapping(condition=emap[0], table_name=emap[1])
            for emap in mappings
        )
        store.close()
        return RamaDBFromText(tables=tables, evaluation_mappings=eval_mappings)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CompactedRamaDatabase:
    _from_rama_db_cache = {}

    table: Tensor(torch.float)[:, 36, 36]
    bspline: BSplineInterpolation
    mapper: RamaMapper

    @classmethod
    @toolz.functoolz.memoize(
        cache=_from_rama_db_cache,
        key=lambda args, kwargs: (args[1], args[2].type, args[2].index),
    )
    def from_ramadb(cls, ramadb: RamaDatabase, device: torch.device):
        """
        Construct a CompactedRamaDatabase from a RamaDatabase.

        Ensure only one compacted copy of the database is created for either
        the CPU or the GPU by using a memoization of the device and the RamaDatabase;
        The RamaDatabase is hashed based on the name of the file that was used
        to create it.
        """

        table = torch.full(
            (len(ramadb.tables), 36, 36), -1234, dtype=torch.float, device=device
        )

        for i, tab in enumerate(ramadb.tables):
            n_rows = int(360 // tab.psi_step)
            n_cols = 360 // tab.phi_step
            assert tab.probabilities.shape[0] == n_rows
            assert tab.probabilities.shape[1] == n_cols

            # effectively reshaping the matrix from the human-readable,
            # if questionably laid out, format that resembles the
            # X and Y axis of the Ramachandran plot (where the upper-left
            # corner is the (phi=-180,psi=+180) coordinate), to one where
            # [0,0] refers to phi=-180, psi=-180.
            for j in range(tab.probabilities.shape[0]):
                for k in range(tab.probabilities.shape[1]):
                    phi_ind = k
                    psi_ind = n_rows - j - 1
                    table[i, phi_ind, psi_ind] = tab.probabilities[j, k]

        # exp of the -energies should get back to the original probabilities
        # so we can calculate the table entropies
        entropy = (
            ((table * torch.log(table)).sum(dim=2))
            .sum(dim=1)
            .reshape(len(ramadb.tables), 1, 1)
        )
        table = -1 * torch.log(table) + entropy

        bspline = BSplineInterpolation.from_coordinates(table, degree=3, n_index_dims=1)

        return cls(table=table, bspline=bspline, mapper=ramadb.mapper)
