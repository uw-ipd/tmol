import numpy
import torch
import attr
from collections import defaultdict

from tmol.database import ParameterDatabase
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase
from typing import Tuple, Mapping  # , FrozenSet
from .pdb_parsing import parse_pdb
import toolz.functoolz

# ordered_canonical_aa_types = (
#     "ALA",
#     "CYS",
#     "ASP",
#     "GLU",
#     "PHE",
#     "GLY",
#     "HIS",
#     "ILE",
#     "LYS",
#     "LEU",
#     "MET",
#     "ASN",
#     "PRO",
#     "GLN",
#     "ARG",
#     "SER",
#     "THR",
#     "VAL",
#     "TRP",
#     "TYR",
# )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class CysSpecialCaseIndices:
    cys_co_aa_ind: int
    sg_atom_for_co_cys: int


@attr.s(slots=True, frozen=True)
class HisSpecialCaseIndices:
    his_co_aa_ind: int = attr.ib()
    his_ND1_in_co: int = attr.ib()
    his_NE2_in_co: int = attr.ib()
    his_HD1_in_co: int = attr.ib()
    his_HE2_in_co: int = attr.ib()
    his_HN_in_co: int = attr.ib()
    his_NH_in_co: int = attr.ib()
    his_NN_in_co: int = attr.ib()
    his_CG_in_co: int = attr.ib()
    _hash: int = attr.ib()

    @_hash.default
    def _init_hash(self):
        return hash(
            (
                self.his_co_aa_ind,
                self.his_ND1_in_co,
                self.his_NE2_in_co,
                self.his_HD1_in_co,
                self.his_HE2_in_co,
                self.his_HN_in_co,
                self.his_NH_in_co,
                self.his_NN_in_co,
                self.his_CG_in_co,
            )
        )

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self._hash == other._hash


@attr.s(auto_attribs=True, frozen=True)
class CanonicalOrdering:
    restype_io_equiv_classes: Tuple[str, ...]
    # the name for each atom
    restypes_ordered_atom_names: Mapping[str, Tuple[str, ...]]
    # mapping for each name3 from atom name and alternate atom name to canonical form index
    restypes_atom_index_mapping: Mapping[str, Mapping[str, int]]
    restypes_default_termini_mapping: Mapping[str, Tuple[str, str]]
    down_termini_patches: Tuple[str, ...]
    up_termini_patches: Tuple[str, ...]
    termini_patch_added_atoms: Mapping[str, Tuple[str, ...]]
    cys_inds: CysSpecialCaseIndices
    his_inds: HisSpecialCaseIndices
    max_n_canonical_atoms: int

    @property
    def n_restype_io_equiv_classes(self):
        return len(self.restype_io_equiv_classes)

    @classmethod
    def extra_atoms(cls):
        return {
            "HIS": ["NN", "NH", "HN"]
            # to do: "CYS": ["HGT"]
        }

    @classmethod
    def from_chemdb(cls, chemdb: PatchedChemicalDatabase):
        class ordered_set:
            def __init__(self, input_values=None):
                self.ordered_vals = []
                self.unordered_vals = set([])
                if input_values is not None:
                    for val in input_values:
                        self.add(val)

            def add(self, val):
                if val not in self.unordered_vals:
                    self.unordered_vals.add(val)
                    self.ordered_vals.append(val)

        restypes = ordered_set(rt.io_equiv_class for rt in chemdb.residues)
        ordered_restypes = restypes.ordered_vals

        def newset():
            return ordered_set()

        restypes_all_atom_names = defaultdict(newset)
        restypes_alt_atom_name_mapping = defaultdict(dict)

        for restype in chemdb.residues:
            for at in restype.atoms:
                restypes_all_atom_names[restype.name3].add(at.name)
            for at in restype.atom_aliases:
                if at.alt_name in restypes_alt_atom_name_mapping[restype.name3]:
                    assert (
                        restypes_alt_atom_name_mapping[restype.name3][at.alt_name]
                        == at.name
                    )
                else:
                    restypes_alt_atom_name_mapping[restype.name3][at.alt_name] = at.name

        # note that extra atoms are internal-use only and do not have "alternate" names
        extra = cls.extra_atoms()
        for rt_name3, atoms in extra.items():
            for at in atoms:
                restypes_all_atom_names[rt_name3].add(at)

        restypes_ordered_atom_names = {
            name3: ats.ordered_vals for name3, ats in restypes_all_atom_names.items()
        }
        restypes_atom_index_mapping = {
            name3: {at: i for i, at in enumerate(ordered_atoms)}
            for name3, ordered_atoms in restypes_ordered_atom_names.items()
        }
        for name3, mapping in restypes_alt_atom_name_mapping.items():
            for alt_name, name in mapping.items():
                restypes_atom_index_mapping[name3][
                    alt_name
                ] = restypes_atom_index_mapping[name3][name]

        max_n_canonical_atoms = max(
            len(atoms) for _, atoms in restypes_ordered_atom_names.items()
        )

        # base_restypes_atom_index_mapping = {}
        # for name3 in ordered_restypes:
        #     atom_indices = {}
        #     for i, at in enumerate(restypes_ordered_atom_names[name3]):
        #         atom_indices[at] = i
        #     for i, at in enumerate(restypes_ordered_alt_atom_names[name3]):
        #         if at not in atom_indices:
        #             atom_indices[at] = i
        #     restypes_atom_index_mapping[bt] = atom_indices

        default_termini_mapping = cls._temp_termini_mapping()
        default_termini_variants = set(
            [
                x
                for _, up_down_tuple in default_termini_mapping.items()
                for x in up_down_tuple
                if x != ""
            ]
        )
        # up_termini_variant_added_atoms = defaultdict(lambda: set([]))
        termini_patch_added_atoms = defaultdict(lambda: set([]))

        # we need to know which variants create down- and up termini
        # so we can build the right termini types
        down_termini_patches = set([])
        up_termini_patches = set([])
        for patch in chemdb.variants:
            for rm in patch.remove_atoms:
                if rm == "<{down}>":
                    down_termini_patches.add(patch.display_name)
                    for atom in patch.add_atoms:
                        termini_patch_added_atoms[patch.display_name].add(atom.name)
                elif rm == "<{up}>":
                    up_termini_patches.add(patch.display_name)
                    for atom in patch.add_atoms:
                        termini_patch_added_atoms[patch.display_name].add(atom.name)

        return cls(
            restype_io_equiv_classes=ordered_restypes,
            restypes_ordered_atom_names=restypes_ordered_atom_names,
            restypes_atom_index_mapping=restypes_atom_index_mapping,
            restypes_default_termini_mapping=default_termini_mapping,
            down_termini_patches=down_termini_patches,
            up_termini_patches=up_termini_patches,
            termini_patch_added_atoms=termini_patch_added_atoms,
            cys_inds=cls._init_cys_special_case_indices(
                ordered_restypes, restypes_ordered_atom_names
            ),
            his_inds=cls._init_his_special_case_indices(
                ordered_restypes, restypes_ordered_atom_names
            ),
            max_n_canonical_atoms=max_n_canonical_atoms,
        )

    @classmethod
    def _temp_termini_mapping(cls):
        return {
            "ALA": ("nterm", "cterm"),
            "CYS": ("nterm", "cterm"),
            "CYD": ("nterm", "cterm"),
            "ASP": ("nterm", "cterm"),
            "GLU": ("nterm", "cterm"),
            "PHE": ("nterm", "cterm"),
            "GLY": ("nterm", "cterm"),
            "HIS": ("nterm", "cterm"),
            "HIS_D": ("nterm", "cterm"),
            "ILE": ("nterm", "cterm"),
            "LYS": ("nterm", "cterm"),
            "LEU": ("nterm", "cterm"),
            "MET": ("nterm", "cterm"),
            "ASN": ("nterm", "cterm"),
            "PRO": ("nterm", "cterm"),
            "GLN": ("nterm", "cterm"),
            "ARG": ("nterm", "cterm"),
            "SER": ("nterm", "cterm"),
            "THR": ("nterm", "cterm"),
            "VAL": ("nterm", "cterm"),
            "TRP": ("nterm", "cterm"),
            "TYR": ("nterm", "cterm"),
        }

    @classmethod
    def _init_cys_special_case_indices(
        cls, restype_name3s, restypes_ordered_atom_names
    ):
        cys_co_aa_ind = restype_name3s.index("CYS")
        if cys_co_aa_ind != -1:
            return CysSpecialCaseIndices(
                cys_co_aa_ind=cys_co_aa_ind,
                sg_atom_for_co_cys=restypes_ordered_atom_names["CYS"].index("SG"),
            )
        else:
            return CysSpecialCaseIndices(
                cys_co_aa_ind=cys_co_aa_ind,
                sg_atom_for_co_cys=-1,
            )

    @classmethod
    def _init_his_special_case_indices(
        cls, restype_name3s, restypes_ordered_atom_names
    ):
        his_co_aa_ind = restype_name3s.index("HIS")
        if his_co_aa_ind == -1:
            return HisSpecialCaseIndices(
                his_co_aa_ind=-1,
                his_ND1_in_co=-1,
                his_NE2_in_co=-1,
                his_HD1_in_co=-1,
                his_HE2_in_co=-1,
                his_HN_in_co=-1,
                his_NH_in_co=-1,
                his_NN_in_co=-1,
                his_CG_in_co=-1,
            )
        else:

            def his_at_ind(atname):
                return restypes_ordered_atom_names["HIS"].index(atname)

            return HisSpecialCaseIndices(
                his_co_aa_ind=his_co_aa_ind,
                his_ND1_in_co=his_at_ind("ND1"),
                his_NE2_in_co=his_at_ind("NE2"),
                his_HD1_in_co=his_at_ind("HD1"),
                his_HE2_in_co=his_at_ind("HE2"),
                his_HN_in_co=his_at_ind("HN"),
                his_NH_in_co=his_at_ind("NH"),
                his_NN_in_co=his_at_ind("NN"),
                his_CG_in_co=his_at_ind("CG"),
            )


@toolz.functoolz.memoize
def default_canonical_ordering():
    chemdb = ParameterDatabase.get_default().chemical
    return CanonicalOrdering.from_chemdb(chemdb)


def default_canonical_form_from_pdb_lines(pdb_lines, device):
    canonical_ordering = default_canonical_ordering()
    return canonical_form_from_pdb_lines(canonical_ordering, pdb_lines, device)


def canonical_form_from_pdb_lines(
    canonical_ordering: CanonicalOrdering, pdb_lines: str, device: torch.device
):
    max_n_canonical_atoms = canonical_ordering.max_n_canonical_atoms
    atom_records = parse_pdb(pdb_lines)
    uniq_res_ind = {}
    uniq_res_list = []
    count_uniq = -1
    for i, row in atom_records.iterrows():
        resid = (row["chain"], row["resi"], row["insert"])
        if resid not in uniq_res_ind:
            count_uniq += 1
            uniq_res_ind[resid] = count_uniq
            uniq_res_list.append(resid)
    n_res = len(uniq_res_list)

    chain_id = numpy.zeros((1, n_res), dtype=numpy.int32)
    res_types = numpy.full((1, n_res), -2, dtype=numpy.int32)
    coords = numpy.full(
        (1, n_res, max_n_canonical_atoms, 3), numpy.NAN, dtype=numpy.float32
    )
    atom_is_present = numpy.zeros((1, n_res, max_n_canonical_atoms), dtype=numpy.bool)

    chains_seen = {}
    chain_id_counter = 0  # TO DO: determine if this is wholly redundant w/ "chaini"
    for i, row in atom_records.iterrows():
        resid = (row["chain"], row["resi"], row["insert"])
        res_ind = uniq_res_ind[resid]
        if row["chaini"] not in chains_seen:
            chains_seen[row["chaini"]] = chain_id_counter
            chain_id_counter += 1
        chain_id[0, res_ind] = chains_seen[row["chaini"]]
        if res_types[0, res_ind] == -2:
            try:
                aa_ind = canonical_ordering.restype_io_equiv_classes.index(row["resn"])
                # print(i, 'row["resn"]', row["resn"], aa_ind)
                res_types[0, res_ind] = aa_ind
            except KeyError:
                res_types[0, res_ind] = -1
        if res_types[0, res_ind] >= 0:
            res_at_mapping = canonical_ordering.restypes_atom_index_mapping[row["resn"]]

            atname = row["atomn"].strip()
            try:
                atind = res_at_mapping[atname]
                atom_is_present[0, res_ind, atind] = True
                coords[0, res_ind, atind, 0] = row["x"]
                coords[0, res_ind, atind, 1] = row["y"]
                coords[0, res_ind, atind, 2] = row["z"]
            except KeyError:
                # ignore atoms that are not in the canonical form
                # TO DO: warn the user that some atoms are not being processed?
                pass

    def _ti32(x):
        return torch.tensor(x, dtype=torch.int32, device=device)

    def _tf32(x):
        return torch.tensor(x, dtype=torch.float32, device=device)

    return (
        _ti32(chain_id),
        _ti32(res_types),
        _tf32(coords),
        torch.tensor(atom_is_present, dtype=torch.bool, device=device),
    )
