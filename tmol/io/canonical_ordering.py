import numpy
import torch

from .pdb_parsing import parse_pdb
import toolz.functoolz

ordered_canonical_aa_types = (
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
)


@attr.s(auto_attrib=True, frozen=True)
class CanonicalOrdering:
    restype_name3s: Tuple[str, ...]
    # the name for each atom
    restypes_ordered_atom_names: Mapping[str, Tuple[str, ...]]
    # mapping for each name3 from atom name and alternate atom name to canonical form index
    restypes_atom_index_mapping: Mapping[str, Mapping[str, int]]
    max_n_canonical_atoms: int

    @classmethod
    def extra_atoms():
        return {"HIS": ["NN", "NH", "HN"]}

    @classmethod
    def from_chemdb(cls, chem_db: ChemicalDatabase):
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

        restypes = ordered_set(rt.name3 for rt in chem_db.residues)
        ordered_restypes = restypes.ordered_vals

        def newset():
            return ordered_set()

        restypes_all_atom_names = defaultdict(newset)
        restypes_all_alt_atom_names = defaultdict(newset)

        # algorithm for getting a mapping from atom names and alternate atom names
        # to the same indices: construct two ordered lists of atom names and make
        # sure to add entries to them in the same order; this will guarantee that
        # they are ordered the same
        for restype in chem_db.residues:
            for at in restype.atoms:
                restypes_all_atom_names[restype.name3].add(at.name)
                restypes_all_alt_atom_names[restype.name3].add(
                    at.name if at.alt_name == "" else at.alt_name
                )

        # note that extra atoms are internal-use only and do not have "alternate" names
        extra = extra_atoms()
        for rt_name3, atoms in extra.items():
            for at in atoms:
                restypes_all_atom_names[rt_name3].add(at)

        restypes_ordered_atom_names = {
            name3: ats.ordered_vals for name3, ats in restypes_all_atom_names.items()
        }
        restypes_ordered_alt_atom_names = {
            name3: ats.ordered_vals
            for name3, ats in restypes_all_alt_atom_names.items()
        }

        max_n_canonical_atoms = max(
            len(atoms) for _, atoms in restypes_ordered_atom_names.items()
        )

        base_restypes_atom_index_mapping = {}
        for name3 in ordered_restypes:
            atom_indices = {}
            for i, at in enumerate(restypes_ordered_atom_names[name3]):
                atom_indices[at] = i
            for i, at in enumerate(restypes_ordered_alt_atom_names[name3]):
                if at not in atom_indices:
                    atom_indices[at] = i
            restypes_atom_index_mapping[bt] = atom_indices

        return cls(
            restype_name3s=ordered_restypes,
            restypes_ordered_atom_names=restypes_ordered_atom_names,
            restypes_atom_index_mapping=restypes_atom_index_mapping,
            max_n_canonical_atoms=max_n_canonical_atoms,
        )


#
# max_n_canonical_atoms = max(
#     len(atlist) for _, atlist in ordered_canonical_aa_atoms.items()
# )


@toolz.functoolz.memoize
def default_chemical_ordering():
    chem_db = ParameterDatabase.get_default().chemical
    return ChemicalOrdering


def default_canonical_form_from_pdb_lines(pdb_lines, device):
    chem_ord = default_chemical_ordering()
    return canonical_form_from_pdb_lines(pdb_lines, device, chemical_ordering)


def canonical_form_from_pdb_lines(pdb_lines, device, chemical_ordering):
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
    atom_is_present = numpy.zeros((1, n_res, max_n_canonical_atoms), dtype=numpy.int32)

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
                aa_ind = ordered_canonical_aa_types.index(row["resn"])
                res_types[0, res_ind] = aa_ind
            except KeyError:
                res_types[0, res_ind] = -1
        if res_types[0, res_ind] >= 0:
            res_at_list = chemical_ordering.base_restypes_atom_index_mapping[
                row["resn"]
            ]

            atname = row["atomn"].strip()
            try:
                atind = res_at_list.index(atname)
                atom_is_present[0, res_ind, atind] = 1
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

    return _ti32(chain_id), _ti32(res_types), _tf32(coords), _ti32(atom_is_present)
