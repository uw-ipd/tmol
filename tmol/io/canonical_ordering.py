import numpy
import torch
import attr
import pandas
from collections import defaultdict

from tmol.types.functional import validate_args
from tmol.database import ParameterDatabase
from tmol.pose.packed_block_types import PackedBlockTypes
from tmol.chemical.patched_chemdb import PatchedChemicalDatabase
from typing import List, Mapping, Optional, Tuple, Union
from .pdb_parsing import parse_pdb
import toolz.functoolz


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
    """The canonical ordering class describes the integer ordering
    of residue types and for atoms within those residue types
    for the collection of available residue types defined by a
    PatchedChemicalDatabase.

    The canonical ordering class's purpose is to enable creation of
    a "canonical form" dictionary that describes a molecular system
    in the way that tmol expects in order to construct a PoseStack.

    There is no "canonical form" dictionary is simply a dictionary
    holding the at-least-three-but-as-many-as-eight arguments to
    tmol.io.pose_stack_construction.pose_stack_from_canonical_form
    after the first two. That is, it must contain "chain_id",
    "res_types" and "coords" entries.

    When constructing a PoseStack, there are multiple residue types
    for each "equivalence class" (think 3-letter code); e.g. for
    "CYS" there's the standard middle-of-a-polypeptide-chain CYS,
    the standard middle-of-a-polypeptide-chain disulfide-forming CYS,
    and then for those two, four variants for the N-, C-, and both-N-
    and-C terminal forms; eight total options for a single "CYS"
    three-letter code. tmol collects all of the various forms of
    a single equivalence class and creates a list of all atom names
    across all the residue types for it. You can then provide tmol
    the set of atoms that are present at a given position by giving
    a non-NaN coordinate for that entry in an
    [n-poses x max-n-res x max-ats-per-res x 3] tensor of
    coordinates. Atoms with NaN coordinates are taken as possibly
    present in the residue type; tmol will decide the best fit
    for which residue type to use at each position.
    If an atom is provided to tmol and it is not present for a
    given residue type, then that residue type will be disqualified
    from consideration. Thus an important part of telling
    tmol which atoms are present is mapping from an atom name to
    an index for that atom. The CanonicalOrdering object is where
    that mapping is encoded. It also handles the mapping from
    alternate-atom-name to canonical-form-atom index; e.g. in
    PDBv2, glycine's two hydrogens were named "HA1" and "HA2",
    but in PDBv3, they are named "1HA" and "2HA." So that we can
    parse PDB files written in PDBv2 and PDBv3, we have an idea of
    an "alias" for an atom; see the restypes_atom_index_mapping
    data member.

    There are four data members that are useful for users:
        - max_n_canonical_atoms
        - restype_io_equiv_classes
        - restypes_ordered_atom_names
        - restypes_atom_index_mapping
    the remaining data members are useful primarily for
    internal tmol functionality

    max_n_canonical_atoms: the largest number of distinct atom names among all
        variants of a single residue type (equivalence class) across all residue types

    restype_io_equiv_classes:
        essentially the list of 3-letter codes for the residue
        types that are readable; use the index function
        (e.g. co.restype_io_equiv_classes.index("TRP"))
        to obtain the integer meant to represent each restype

    restypes_ordered_atom_names:
        the ordered list of the names of each atom for every allowed
        residue type; does not include the alternate names for atoms.
        Atoms should be given to tmol in this order; e.g. by putting
        the coordinate of the ith atom in the ith entry of the
        coordinate tensor (e.g. coords[p, r, i] for pose p, residue r)

    restypes_atom_index_mapping:
        mapping for each name3 from atom name and atom name alias
        to the index of that atom for every allowed residue
        type in the restypes_ordered_atom_names list; this is
        probably more useful than the restypes_ordered_atom_names
        list, especially if you are using the PDBv2 naming
        convention (as Rosetta3 does) instead of the PDBv3
        convention.
    """

    max_n_canonical_atoms: int
    restype_io_equiv_classes: Tuple[str, ...]
    restypes_ordered_atom_names: Mapping[str, Tuple[str, ...]]
    restypes_atom_index_mapping: Mapping[str, Mapping[str, int]]

    ############# tmol internal data members below ############

    restypes_default_termini_mapping: Mapping[str, Tuple[str, str]]
    down_termini_patches: Tuple[str, ...]
    up_termini_patches: Tuple[str, ...]
    termini_patch_added_atoms: Mapping[str, Tuple[str, ...]]
    cys_inds: CysSpecialCaseIndices
    his_inds: HisSpecialCaseIndices

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

        default_termini_mapping = cls._temp_termini_mapping()
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
            max_n_canonical_atoms=max_n_canonical_atoms,
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
        if "CYS" not in restype_name3s:
            return CysSpecialCaseIndices(
                cys_co_aa_ind=-1,
                sg_atom_for_co_cys=-1,
            )
        else:
            cys_co_aa_ind = restype_name3s.index("CYS")
            return CysSpecialCaseIndices(
                cys_co_aa_ind=cys_co_aa_ind,
                sg_atom_for_co_cys=restypes_ordered_atom_names["CYS"].index("SG"),
            )

    @classmethod
    def _init_his_special_case_indices(
        cls, restype_name3s, restypes_ordered_atom_names
    ):
        if "HIS" not in restype_name3s:
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
            his_co_aa_ind = restype_name3s.index("HIS")

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

    def create_src_2_tmol_mappings(
        self, src_aa_name3s, src_atom_names_for_name3s, device
    ):
        src_2_tmol_restype_mapping = torch.full(
            (len(src_aa_name3s),),
            -1,
            dtype=torch.int64,
        )

        # how many atoms does the src format support?
        # each entry in the src_atom_names_for_name3s should be
        # the same length, so we will just ask what the length is
        # of the first name3's atom list
        src_max_n_ats = len(src_atom_names_for_name3s[src_aa_name3s[0]])

        src_2_tmol_atom_mapping = torch.full(
            (len(src_aa_name3s), src_max_n_ats),
            -1,
            dtype=torch.int64,
        )
        src_at_is_real = torch.zeros(
            (len(src_aa_name3s), src_max_n_ats), dtype=torch.bool
        )

        for i, i_3lc in enumerate(src_aa_name3s):
            if i_3lc not in self.restype_io_equiv_classes:
                # Map this RT whenever encountered to a place-holder residue
                continue
            src_2_tmol_restype_mapping[i] = self.restype_io_equiv_classes.index(i_3lc)
            src_res_atoms = src_atom_names_for_name3s[i_3lc]

            # restypes_atom_index_mapping supports atom aliasing
            # which resolves any ambiguity in PDB naming conventions;
            tmol_res_atom_inds = self.restypes_atom_index_mapping[i_3lc]
            for j, at in enumerate(src_res_atoms):
                if at == "":
                    continue
                if at not in tmol_res_atom_inds:
                    raise ValueError(f"error: {i_3lc} atom {at} not in tmol atom set")
                src_2_tmol_atom_mapping[i, j] = tmol_res_atom_inds[at]
                src_at_is_real[i, j] = True

        def _d(x):
            return x.to(device=device)

        return (
            _d(src_2_tmol_restype_mapping),
            _d(src_2_tmol_atom_mapping),
            _d(src_at_is_real),
        )


@validate_args
@toolz.functoolz.memoize
def default_canonical_ordering() -> CanonicalOrdering:
    """Create a CanonicalOrdering object from the default set of residue types"""

    chemdb = ParameterDatabase.get_default().chemical
    return CanonicalOrdering.from_chemdb(chemdb)


@validate_args
@toolz.functoolz.memoize
def default_packed_block_types(device: torch.device) -> PackedBlockTypes:
    """Create a PackedBlockTypes object from the default set of residue types"""
    import cattr
    from tmol.chemical.restypes import RefinedResidueType
    from tmol.pose.packed_block_types import PackedBlockTypes

    chem_database = ParameterDatabase.get_default().chemical

    restype_list = [
        cattr.structure(
            cattr.unstructure(r),
            RefinedResidueType,
        )
        for r in chem_database.residues
    ]

    return PackedBlockTypes.from_restype_list(chem_database, restype_list, device)


@validate_args
def canonical_form_from_pdb(
    canonical_ordering: CanonicalOrdering,
    pdb_lines_or_fname: Union[str, List],
    device: torch.device,
    *,
    residue_start: Optional[int] = None,
    residue_end: Optional[int] = None,
) -> Mapping:
    """Create a canonical form dictionary from either the contents of a PDB file
    as one long string or a list of individual lines from the file or
    by providing the name/path of a PDB file

    pdb_lines_or_fname must either be a list of the lines in a PDB file or
    a string representing a file

    """
    atom_records = parse_pdb(pdb_lines_or_fname)
    if residue_start is not None or residue_end is not None:
        atom_records = select_atom_records_res_subset(
            atom_records, residue_start, residue_end
        )
    return canonical_form_from_atom_records(canonical_ordering, atom_records, device)


def select_atom_records_res_subset(
    atom_records: pandas.DataFrame,
    residue_start: Optional[int],
    residue_end: Optional[int],
):
    """Figure out the starting row index for each residue
    and take the slice of the atom_records dataframe containing
    every atom of every residue within the given inclusive range.
    If either residue_start or residue_end are omitted, then
    the are treated as being the first or last residue.
    """

    atom_records_begin_for_res = []
    last_res = None
    for i, row in atom_records.iterrows():
        this_res = (row["modeli"], row["chaini"], row["resi"])
        if last_res is not None and last_res == this_res:
            pass
        else:
            atom_records_begin_for_res.append(i)
            last_res = this_res
    atom_records_begin_for_res.append(i + 1)
    if residue_start is None:
        residue_start = 0
    if residue_end is None:
        residue_end = len(atom_records_begin_for_res) - 1
    begin = atom_records_begin_for_res[residue_start]
    end = atom_records_begin_for_res[residue_end]
    return atom_records.iloc[begin:end]


def canonical_form_from_atom_records(
    canonical_ordering: CanonicalOrdering,
    atom_records: pandas.DataFrame,
    device: torch.device,
):
    max_n_canonical_atoms = canonical_ordering.max_n_canonical_atoms

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
                res_types[0, res_ind] = aa_ind
            except KeyError:
                res_types[0, res_ind] = -1
        if res_types[0, res_ind] >= 0:
            res_at_mapping = canonical_ordering.restypes_atom_index_mapping[row["resn"]]

            atname = row["atomn"].strip()
            try:
                atind = res_at_mapping[atname]
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

    return dict(
        chain_id=_ti32(chain_id),
        res_types=_ti32(res_types),
        coords=_tf32(coords),
    )
