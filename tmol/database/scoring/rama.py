import attr
import cattr
import numpy
import torch
import yaml
import zarr

from typing import Tuple

from frozendict import frozendict
from tmol.types.torch import Tensor
from tmol.types.functional import validate_args


def safe_fetch_from_zarr(zgroup, array_name):
    numpy_array = numpy.empty(zgroup[array_name].shape, zgroup[array_name].dtype)
    zgroup[array_name].get_basic_selection(..., out=numpy_array)
    return numpy_array


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaTable:
    name: str
    bb_start: Tensor(float)[:]
    bb_step: Tensor(float)[:]
    probabilities: Tensor(float)

    @classmethod
    def from_zarr(cls, zgroup, name):
        table_group = zgroup[name]
        bb_start = torch.tensor(
            safe_fetch_from_zarr(table_group, "bb_start"), dtype=torch.float
        )
        bb_step = torch.tensor(
            safe_fetch_from_zarr(table_group, "bb_step"), dtype=torch.float
        )
        probs = torch.tensor(
            safe_fetch_from_zarr(table_group, "probabilities"), dtype=torch.float
        )
        return cls(name=name, bb_start=bb_start, bb_step=bb_step, probabilities=probs)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class EvaluationMapping:
    condition: str
    table_name: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class EvaluationMappings:
    mappings: Tuple[EvaluationMapping, ...]


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
    evaluation_mappings: EvaluationMappings
    mapper: RamaMapper

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "RamaDatabase(%s)" % self.sourcefilepath

    @classmethod
    def from_files(cls, path):
        store = zarr.LMDBStore(path + ("" if path[-1] == "/" else "/") + "rama.bin")
        zgroup = zarr.group(store)
        table_list = zgroup.attrs["tables"]
        tables = []
        for table_name in table_list:
            tables.append(RamaTable.from_zarr(zgroup, table_name))
        tables = tuple(tables)
        store.close()

        # load the evaluation mappings from yaml
        with open(path + "rama_mapping.yaml") as fid:
            raw = yaml.load(fid, yaml.CLoader)
        evaluation_mappings = cattr.structure(raw, EvaluationMappings)

        mapper = RamaMapper.from_eval_mapping_and_table_list(
            evaluation_mappings.mappings, tables
        )
        return cls(
            sourcefilepath=path,
            tables=tables,
            evaluation_mappings=evaluation_mappings,
            mapper=mapper,
        )
