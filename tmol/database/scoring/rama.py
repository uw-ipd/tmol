import attr
import cattr
import numpy
import os
import re
import torch
import yaml
import zarr

from typing import Tuple

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
class EvaluationCondition:
    central_res: str
    upper_res: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class EvaluationMapping:
    condition: EvaluationCondition
    table_name: str


@attr.s(auto_attribs=True, frozen=True, slots=True)
class EvaluationMappings:
    mappings: Tuple[EvaluationMapping, ...]


def crange(start, stop):
    """Character range

    Expand a pair of ascii characters into an explicit list
    of all characters in between and including."""
    return [chr(c) for c in range(ord(start), ord(stop) + 1)]


def parse_property_pattern(pat):
    """Convert a pattern for identifying residues by properties
    into a python regular expression.

    The pattern
        aa.alpha.l.serine
    will match
        - aa.alpha.l.serine
    as well as
        - aa.alpha.l.serine.deprotonated
        - aa.alpha.l.serine.phosphorylated
    but will not match
        - aa.alpha.l.threonine
    or
        - aa.alpha.l.serineoicacid

    The idea is that anything after the leading pattern
    is acceptible so long as there is a period separating
    the leading pattern from the rest of the property.
    """

    invert = False
    if pat[0] == "!":
        pat = pat[1:]
        invert = True

    allowed_chars = set(
        crange("a", "z") + crange("A", "Z") + crange("0", "9") + ["_", "-", "[", "]"]
    )

    result = []

    for pchar in pat:
        if pchar == "*":
            result.append(r"[a-zA-Z0-9_\-\[\]]*")
        elif pchar == ".":
            result.append(r"\.")
        elif pchar in allowed_chars:
            result.append(re.escape(pchar))
        else:
            raise ValueError("Invalid pattern: %r invalid char: %r" % (pat, pchar))

    result.append(r"(\.[\.a-zA-Z0-9_\-\[\]]+)?")
    result.append("$")

    result_re = "".join(result)

    if invert:
        result_re = f"(?!{result_re})"

    result_re = f"^{result_re}"
    return result_re


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaSingleMapper:
    central_res: str
    upper_res: str
    upper_res_pos: bool
    which_table: int
    central_re: str
    upper_re: str

    @staticmethod
    @validate_args
    def from_condition(condition: EvaluationCondition, which_table: int):
        """Turn the condition string in rama.json into the parts that
        decide which table to use for each residue"""
        assert (
            condition.central_res[0] != "!"
        ), "central residue cannot be defined by a negative"
        upper_res_pos = condition.upper_res[0] != "!"

        central_re = parse_property_pattern(condition.central_res)
        upper_re = parse_property_pattern(condition.upper_res)

        return RamaSingleMapper(
            central_res=condition.central_res,
            upper_res=condition.upper_res,
            upper_res_pos=upper_res_pos,
            which_table=which_table,
            central_re=central_re,
            upper_re=upper_re,
        )

    def matches(
        self, cent_res_props: Tuple[str, ...], upper_res_props: Tuple[str, ...]
    ) -> bool:
        cent_matches = False

        # All cent-res conditions must be positive
        # proceed as long as any property matches
        for prop in cent_res_props:
            if re.match(self.central_re, prop):
                cent_matches = True
                break
        if not cent_matches:
            return False

        if self.upper_res_pos:
            # return true if any of the properties match
            for prop in upper_res_props:
                if re.match(self.upper_re, prop):
                    return True
            return False
        else:
            # return true if all of the properties match
            for prop in upper_res_props:
                if not re.match(self.upper_re, prop):
                    return False
            return True


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaMapper:
    """Map properties for residue i and i+1 into a table index

    The amino acid properties for residues i and i+1 will be examined
    to answer the question: "which of the various tables that have been
    defined in the input file should be used for residue i?"
    """

    mappers: Tuple[RamaSingleMapper, ...]

    @validate_args
    def from_eval_mapping_and_table_list(
        evaluation_mappings: Tuple[EvaluationMapping, ...],
        tables: Tuple[RamaTable, ...],
    ):
        mappers = []
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
            mappers.append(RamaSingleMapper.from_condition(ev_map.condition, tab_ind))
        return RamaMapper(mappers=mappers)

    def table_ind_for_res(
        self, cent_res_props: Tuple[str, ...], upper_res_props: Tuple[str, ...]
    ) -> int:
        """
        Return the index of the table to use for the central residue, or -1 if
        there is no table which residue i matches.
        """
        for mapper in self.mappers:
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
        store = zarr.LMDBStore(os.path.join(path, "rama.bin"))
        zgroup = zarr.hierarchy.open_group(store, mode="r")
        table_list = zgroup.attrs["tables"]
        tables = []
        for table_name in table_list:
            tables.append(RamaTable.from_zarr(zgroup, table_name))
        tables = tuple(tables)
        store.close()

        # load the evaluation mappings from yaml
        with open(os.path.join(path, "rama_mapping.yaml")) as fid:
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
