"""Regenerate database/default/scoring/na_torsion.yaml from crystal structures.

Rosetta derives its DNA means at load time by parsing raw observations
(database/scoring/dna/bound_dna_dihedrals.txt); tmol ships the derived means.
This script rebuilds them from a structure list, using Rosetta's binning and
averaging but computing proper nu ring torsions from heavy atoms rather than
Rosetta's substituent-referenced "chi2"/"chi4".

DNA and RNA share the functional form but not the parameters, so both are
swept in one run and written to one file under a dna: and an rna: key. The
structure lists are na_torsion_dna_structures.txt (protein-DNA X-ray entries
at <= 2.3 A, one representative per 30% sequence-identity cluster) and
na_torsion_rna_structures.txt.

    python -m tmol.support.scoring.na_torsion_param_import \
        --pdb-dir ~/na_structures \
        --out tmol/database/default/scoring/na_torsion.yaml

    # rebuild the structure lists from RCSB rather than reusing them
    python -m tmol.support.scoring.na_torsion_param_import --requery

    # check the pipeline against Rosetta's own DNA observations
    python -m tmol.support.scoring.na_torsion_param_import \
        --pdb-dir ~/na_structures --validate ~/rosetta/database/scoring/dna/\
bound_dna_dihedrals.txt
"""

import argparse
import datetime
import gzip
import json
import os
import urllib.request
from collections import defaultdict

import numpy

HERE = os.path.dirname(os.path.abspath(__file__))
DNA_STRUCTURE_LIST = os.path.join(HERE, "na_torsion_dna_structures.txt")
RNA_STRUCTURE_LIST = os.path.join(HERE, "na_torsion_rna_structures.txt")

PURINE = {"DA", "DG", "A", "G", "ADE", "GUA"}
PYRIMIDINE = {"DC", "DT", "C", "T", "CYT", "THY", "U", "URA"}
NA_RESNAMES = PURINE | PYRIMIDINE
BASE1 = {
    "DA": "a", "A": "a", "ADE": "a", "DC": "c", "C": "c", "CYT": "c",
    "DG": "g", "G": "g", "GUA": "g", "DT": "t", "T": "t", "THY": "t",
    "U": "u", "URA": "u",
}  # fmt: skip

# bare A/C/G are the pre-remediation DNA spellings as well as the RNA ones, so
# the sugar decides: only ribose carries O2'. Thymine and uracil are decisive.
POLYMERS = ("dna", "rna")
BASE_ORDER = {"dna": "acgt", "rna": "acgu"}
DEOXY_ONLY = {"DA", "DC", "DG", "DT", "T", "THY"}
RIBO_ONLY = {"U", "URA"}

RING = ["C1'", "C2'", "C3'", "C4'", "O4'"]

# sugar torsion slots, mirroring Rosetta's (delta, chi2, chi3, chi4, chi) but
# with proper literature nu in place of its substituent-referenced torsions
SUGAR_TORSIONS = [
    ("delta", None),  # from the backbone
    ("nu4", ["C3'", "C4'", "O4'", "C1'"]),
    ("nu0", ["C4'", "O4'", "C1'", "C2'"]),
    ("nu1", ["O4'", "C1'", "C2'", "C3'"]),
    ("chi", None),  # glycosidic, from the base
]
N_SUGAR = len(SUGAR_TORSIONS)
N_PUCKER = 10

MIN_TORSIONS = 15  # per bin before borrowing from neighbouring puckers

# well-depth tables: -ln of the bin populations, so a rare rotamer costs more
# than a common one. The pseudocount bounds an empty cell at ln(n_max/a) ~ 14.
PSEUDOCOUNT = 0.01
NORTH_PUCKERS = (0, 1, 2, 3, 4)  # C3'-endo side; the rest are south
MIN_CHI_TORSION = 120.0  # chi below this is syn and excluded from the mean

# Rosetta option defaults (dna::specificity)
SDEV_BACKBONE = [17.0, 30.0, 15.0, 0.0, 20.0, 30.0]
SDEV_SUGAR = 4.0
SDEV_CHI = 15.0

# subterm weights from beta16_opt46A.523; na_torsion sums them at weight 1
WEIGHT_BB = 0.46
WEIGHT_CHI = 1.07
WEIGHT_SUGAR = 0.16

# softmax temperature for the pucker states, and the gaussian width blending
# the three alpha/gamma bins; both replace hard bin assignments
PUCKER_TEMPERATURE = 0.05
BIN_BLEND_SDEV = 30.0

RCSB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_FILES = "https://files.rcsb.org/download/"


# ---------------------------------------------------------------- structures


def query_rcsb(resolution=2.3, seqid_cutoff=30):
    """Representative protein entities of protein-DNA X-ray structures."""

    def text(attribute, operator, value):
        return {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": attribute,
                "operator": operator,
                "value": value,
            },
        }

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                text(
                    "rcsb_entry_info.resolution_combined",
                    "less_or_equal",
                    resolution,
                ),
                text("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
                text(
                    "rcsb_entry_info.polymer_entity_count_DNA",
                    "greater_or_equal",
                    1,
                ),
                text("entity_poly.rcsb_entity_polymer_type", "exact_match", "Protein"),
            ],
        },
        "return_type": "polymer_entity",
        "request_options": {
            "group_by": {
                "aggregation_method": "sequence_identity",
                "similarity_cutoff": seqid_cutoff,
            },
            "group_by_return_type": "representatives",
            "paginate": {"start": 0, "rows": 10000},
            "results_verbosity": "compact",
        },
    }
    req = urllib.request.Request(
        RCSB_SEARCH,
        data=json.dumps(query).encode(),
        headers={"Content-Type": "application/json"},
    )
    result = json.loads(urllib.request.urlopen(req, timeout=300).read())
    ids = [r if isinstance(r, str) else r["identifier"] for r in result["result_set"]]
    return sorted({i.split("_")[0].lower() for i in ids})


def fetch(code, pdb_dir):
    """Local path to a structure, downloading it if absent."""
    for ext in (".pdb.gz", ".cif.gz"):
        path = os.path.join(pdb_dir, code + ext)
        if os.path.exists(path) and os.path.getsize(path):
            return path
    os.makedirs(pdb_dir, exist_ok=True)
    for ext in (".pdb.gz", ".cif.gz"):
        path = os.path.join(pdb_dir, code + ext)
        try:
            urllib.request.urlretrieve(RCSB_FILES + code.upper() + ext, path)
            if os.path.getsize(path):
                return path
        except Exception:
            if os.path.exists(path):
                os.remove(path)
    return None


# ---------------------------------------------------------------------- i/o


def _norm(name):
    """Pre-remediation PDB files spell primes as stars."""
    return name.strip().replace("*", "'")


def read_pdb(path, max_b=None):
    """-> [(chain, resid, resname, {atom: xyz})] for DNA residues, in order."""
    opener = gzip.open if path.endswith(".gz") else open
    out, cur, key = [], None, None
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue
            if line[17:20].strip() not in NA_RESNAMES:
                continue
            name = _norm(line[12:16])
            k = (line[21], line[22:27])
            if k != key:
                cur = {}
                out.append((k[0], k[1], line[17:20].strip(), cur))
                key = k
            if name in cur:  # keep the first altloc
                continue
            try:
                b = float(line[60:66])
            except ValueError:
                b = 0.0
            if max_b is not None and b > max_b:
                continue
            cur[name] = numpy.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    return out


def read_cif(path, max_b=None):
    """Same as read_pdb, via biotite (used for entries with no PDB format)."""
    import biotite.structure.io.pdbx as pdbx

    with gzip.open(path, "rt", errors="replace") as f:
        block = pdbx.CIFFile.read(f)
    arr = pdbx.get_structure(
        block, model=1, extra_fields=["b_factor", "occupancy"], altloc="first"
    )
    arr = arr[numpy.isin(arr.res_name, list(NA_RESNAMES))]
    out, cur, key = [], None, None
    for i in range(arr.array_length()):
        k = (str(arr.chain_id[i]), str(arr.res_id[i]) + str(arr.ins_code[i]))
        if k != key:
            cur = {}
            out.append((k[0], k[1], str(arr.res_name[i]), cur))
            key = k
        if max_b is not None and arr.b_factor[i] > max_b:
            continue
        cur.setdefault(_norm(str(arr.atom_name[i])), arr.coord[i].astype(float))
    return out


def read_structure(path, max_b=None):
    return (read_cif if path.endswith(".cif.gz") else read_pdb)(path, max_b)


# ----------------------------------------------------------------- geometry


def dihedral(p0, p1, p2, p3):
    """Dihedral in [0, 360)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / numpy.linalg.norm(b1)
    v = b0 - (b0 @ b1) * b1
    w = b2 - (b2 @ b1) * b1
    return numpy.degrees(numpy.arctan2(numpy.cross(b1, v) @ w, v @ w)) % 360.0


def subtract_degree_angles(a, b):
    """a - b wrapped to [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def triple_bin(angle):
    """g+/t/g- bin, 1-3, on [0,360)."""
    angle %= 360.0
    return 1 if angle < 120.0 else (2 if angle < 240.0 else 3)


def b1b2_bin(epsilon, zeta):
    """BI (1) or BII (2) from the epsilon-zeta difference."""
    return 1 if subtract_degree_angles(epsilon, zeta) < 0 else 2


def sugar_pucker(atoms):
    """Rosetta's discrete pucker index, 0-9.

    Walks the 5 cyclic rotations of the ring; the apex is the atom left out of
    the most planar four. Endo/exo doubles that to 10 states.
    """
    xyz = [atoms[n] for n in RING]
    names = list(RING)
    mindot, apex, exxo = 1e9, None, False
    for _ in range(5):
        n12 = numpy.cross(xyz[1] - xyz[0], xyz[2] - xyz[1])
        n12 = n12 / numpy.linalg.norm(n12)
        d = abs(n12 @ ((xyz[3] - xyz[2]) / numpy.linalg.norm(xyz[3] - xyz[2])))
        if d < mindot:
            mindot = d
            apex = names[4]
            v = xyz[4] - 0.5 * (xyz[3] + xyz[0])
            exxo = (n12 @ (v / numpy.linalg.norm(v))) > 0.0
        xyz.append(xyz.pop(0))
        names.append(names.pop(0))

    i = RING.index(apex)
    return (i + 1 if i % 2 == (0 if exxo else 1) else i - 4) + 4


def residue_torsions(prev, cur, nxt):
    """alpha..zeta, chi and nu0/nu1/nu4 in [0,360); None where undefined."""
    p_, c_, n_ = (r[3] if r else None for r in (prev, cur, nxt))

    def dih(*names_and_dicts):
        pts = [d.get(n) if d else None for d, n in names_and_dicts]
        return None if any(p is None for p in pts) else dihedral(*pts)

    t = {
        "alpha": dih((p_, "O3'"), (c_, "P"), (c_, "O5'"), (c_, "C5'")),
        "beta": dih((c_, "P"), (c_, "O5'"), (c_, "C5'"), (c_, "C4'")),
        "gamma": dih((c_, "O5'"), (c_, "C5'"), (c_, "C4'"), (c_, "C3'")),
        "delta": dih((c_, "C5'"), (c_, "C4'"), (c_, "C3'"), (c_, "O3'")),
        "epsilon": dih((c_, "C4'"), (c_, "C3'"), (c_, "O3'"), (n_, "P")),
        "zeta": dih((c_, "C3'"), (c_, "O3'"), (n_, "P"), (n_, "O5'")),
    }
    n1, c1 = ("N9", "C4") if cur[2] in PURINE else ("N1", "C2")
    t["chi"] = dih((c_, "O4'"), (c_, "C1'"), (c_, n1), (c_, c1))
    for name, atoms in SUGAR_TORSIONS:
        if atoms is not None:
            t[name] = dih(*[(c_, a) for a in atoms])
    return t


# -------------------------------------------------------------- observations


def polymer_of(resname, atoms):
    """ "dna", "rna", or None if the name and the sugar disagree."""
    ribose = "O2'" in atoms
    if ribose and resname in DEOXY_ONLY:
        return None
    if not ribose and resname in RIBO_ONLY:
        return None
    return "rna" if ribose else "dna"


def observations(codes, pdb_dir, max_b=None, verbose=False):
    """Collect per-nucleotide torsions from every structure in the list."""
    obs, n_struct, n_skipped = [], 0, 0
    for code in codes:
        path = fetch(code, pdb_dir)
        if path is None:
            n_skipped += 1
            continue
        try:
            residues = read_structure(path, max_b)
        except Exception as err:
            if verbose:
                print(f"  {code}: unreadable ({err})")
            n_skipped += 1
            continue
        n_struct += 1
        for i, res in enumerate(residues):
            prev = residues[i - 1] if i > 0 and residues[i - 1][0] == res[0] else None
            nxt = (
                residues[i + 1]
                if i + 1 < len(residues) and residues[i + 1][0] == res[0]
                else None
            )
            if not all(a in res[3] for a in RING):
                continue
            poly = polymer_of(res[2], res[3])
            if poly is None:
                continue
            t = residue_torsions(prev, res, nxt)
            if t["delta"] is None or t["chi"] is None:
                continue
            if any(t[n] is None for n, a in SUGAR_TORSIONS if a is not None):
                continue
            obs.append(
                dict(
                    code=code,
                    poly=poly,
                    base=BASE1[res[2]],
                    pucker=sugar_pucker(res[3]),
                    lower=t["alpha"] is None,
                    upper=t["epsilon"] is None or t["zeta"] is None,
                    tor=t,
                )
            )
    return obs, n_struct, n_skipped


# ------------------------------------------------------------------- means


def circular_mean(values):
    """Rosetta's median-centred mean: robust to the 0/360 seam."""
    values = sorted(values)
    median = values[(len(values) + 1) // 2 - 1]
    offset = numpy.mean([subtract_degree_angles(v, median) for v in values])
    return (median + offset) % 360.0


def sugar_means(obs, bases):
    """mean_sugar_torsion[base][pucker][slot]; only chi is base-dependent."""
    # bucket non-terminal observations by pucker
    by_pucker = defaultdict(lambda: [[] for _ in range(N_SUGAR)])
    inames = defaultdict(list)
    for o in obs:
        if o["lower"] or o["upper"]:
            continue
        for s, (name, _) in enumerate(SUGAR_TORSIONS):
            by_pucker[o["pucker"]][s].append(o["tor"][name])
        inames[o["pucker"]].append(o["base"])

    def pool(pucker, slot, base):
        vals = list(by_pucker[pucker][slot])
        if base is not None:
            keep = inames[pucker]
            vals = [v for v, b in zip(vals, keep) if b == base]
        if len(vals) >= MIN_TORSIONS:
            return vals, len(vals)
        # borrow from neighbouring puckers in widening windows
        have = len(vals)
        window = 0
        while len(vals) < MIN_TORSIONS and window < N_PUCKER:
            window += 1
            for offset in (-window, window):
                # the pucker states are a cycle: 9 neighbours 0
                p = (pucker + offset) % N_PUCKER
                other = list(by_pucker[p][slot])
                if base is not None:
                    other = [v for v, b in zip(other, inames[p]) if b == base]
                vals += other[: MIN_TORSIONS - len(vals)]
        return vals, have

    table = numpy.zeros((4, N_PUCKER, N_SUGAR))
    counts = numpy.zeros((4, N_PUCKER, N_SUGAR), dtype=int)
    for bi, base in enumerate(bases):
        for pucker in range(N_PUCKER):
            for slot, (name, _) in enumerate(SUGAR_TORSIONS):
                seq_dep = name == "chi"
                vals, have = pool(pucker, slot, base if seq_dep else None)
                if not vals:
                    continue
                if seq_dep:  # drop syn conformations
                    vals = sorted(vals)
                    while len(vals) > MIN_TORSIONS - 2 and vals[0] < MIN_CHI_TORSION:
                        vals.pop(0)
                table[bi, pucker, slot] = circular_mean(vals)
                counts[bi, pucker, slot] = have
    return table, counts


def backbone_means(obs):
    """mean_backbone_torsion[tor][bin]; tor 1-6, delta (4) unused."""
    table = numpy.zeros((7, 4))
    counts = numpy.zeros((7, 4), dtype=int)

    def plain_mean(vals):
        return float(numpy.mean(vals)) if vals else 0.0

    interior = [o for o in obs if not o["lower"] and not o["upper"]]

    for tor, name in ((1, "alpha"), (3, "gamma")):
        for b in (1, 2, 3):
            vals = [o["tor"][name] for o in interior if triple_bin(o["tor"][name]) == b]
            table[tor][b], counts[tor][b] = plain_mean(vals), len(vals)

    # beta is binned on the *previous* residue's BI/BII state
    for b in (1, 2):
        vals = []
        for i, o in enumerate(obs):
            if i == 0 or o["lower"] or o["upper"] or obs[i - 1]["lower"]:
                continue
            p = obs[i - 1]["tor"]
            if p["epsilon"] is None or p["zeta"] is None:
                continue
            if b1b2_bin(p["epsilon"], p["zeta"]) == b:
                vals.append(o["tor"]["beta"])
        table[2][b], counts[2][b] = plain_mean(vals), len(vals)

    for tor, name in ((5, "epsilon"), (6, "zeta")):
        for b in (1, 2):
            vals = [
                o["tor"][name]
                for o in interior
                if b1b2_bin(o["tor"]["epsilon"], o["tor"]["zeta"]) == b
            ]
            table[tor][b], counts[tor][b] = plain_mean(vals), len(vals)

    return table, counts


def _offsets(counts, axis=None):
    """-ln P from counts, shifted so the deepest well in the table is 0.

    Normalizing along `axis` gives a conditional distribution; the shift is a
    single constant per table so relative depths are preserved.
    """
    p = counts + PSEUDOCOUNT
    p = p / (p.sum(axis=axis, keepdims=True) if axis is not None else p.sum())
    e = -numpy.log(p)
    return e - e.min()


def well_tables(obs, bases):
    """Bin-population energies, factorized along the selected couplings.

    Each bin assignment is charged once: the sugar torsions all read the pucker
    bin, and beta reads the previous residue's BI/BII, so neither gets a table
    of its own.
    """
    north = set(NORTH_PUCKERS)

    pucker = numpy.zeros(N_PUCKER)
    alpha_gamma = numpy.zeros((3, 3))
    bibii_pucker = numpy.zeros((2, 2))  # BI/BII by north/south
    alphanext_bibii = numpy.zeros((3, 2))  # alpha(i+1) bin by BI/BII
    chi_syn = numpy.zeros((2, N_PUCKER, 4))  # anti/syn by pucker and base

    for o in obs:
        if o["lower"] or o["upper"]:
            continue
        pucker[o["pucker"]] += 1
        t = o["tor"]
        alpha_gamma[triple_bin(t["alpha"]) - 1, triple_bin(t["gamma"]) - 1] += 1
        b = b1b2_bin(t["epsilon"], t["zeta"]) - 1
        bibii_pucker[b, 0 if o["pucker"] in north else 1] += 1
        syn = 0 if t["chi"] % 360.0 >= MIN_CHI_TORSION else 1
        chi_syn[syn, o["pucker"], bases.index(o["base"])] += 1

    for i, o in enumerate(obs):
        if o["lower"] or o["upper"] or i + 1 >= len(obs) or obs[i + 1]["lower"]:
            continue
        nxt = obs[i + 1]["tor"]["alpha"]
        if nxt is None:
            continue
        b = b1b2_bin(o["tor"]["epsilon"], o["tor"]["zeta"]) - 1
        alphanext_bibii[triple_bin(nxt) - 1, b] += 1

    return dict(
        pucker=(_offsets(pucker), pucker),
        alpha_gamma=(_offsets(alpha_gamma), alpha_gamma),
        bibii_given_pucker=(_offsets(bibii_pucker, axis=0), bibii_pucker),
        alphanext_given_bibii=(_offsets(alphanext_bibii, axis=0), alphanext_bibii),
        chi_syn_given_pucker=(_offsets(chi_syn, axis=0), chi_syn),
    )


# -------------------------------------------------------------------- output


def observed_sdev(obs, sugar, bases):
    """Spread of the observations about their own bin mean, per sugar torsion."""
    dev = {name: [] for name, _ in SUGAR_TORSIONS}
    for o in obs:
        if o["lower"] or o["upper"]:
            continue
        b = bases.index(o["base"])
        for s, (name, _) in enumerate(SUGAR_TORSIONS):
            dev[name].append(
                subtract_degree_angles(o["tor"][name], sugar[b, o["pucker"], s])
            )
    return {k: float(numpy.std(v)) for k, v in dev.items() if v}


def _emit_globals(out, sdev_obs):
    """Rosetta's own option table annotates sdev_sugar "## too small"; both are
    tighter than the data, which the subterm weights absorb."""
    vals = [v for k, v in sdev_obs.items() if k != "chi"]
    pooled = sum(vals) / len(vals)
    out.write(f"    sdev_sugar: {SDEV_SUGAR}  # observed spread {pooled:.1f}\n")
    out.write(f"    sdev_chi: {SDEV_CHI}  # observed spread {sdev_obs['chi']:.1f}\n")
    out.write(
        "    sdev_backbone: ["
        + ", ".join(f"{s}" for s in SDEV_BACKBONE)
        + "]  # alpha beta gamma delta epsilon zeta\n"
    )
    out.write(f"    weight_bb: {WEIGHT_BB}\n")
    out.write(f"    weight_chi: {WEIGHT_CHI}\n")
    out.write(f"    weight_sugar: {WEIGHT_SUGAR}\n")
    out.write(f"    pucker_temperature: {PUCKER_TEMPERATURE}\n")
    out.write(f"    bin_blend_sdev: {BIN_BLEND_SDEV}\n")


def _emit_backbone(out, backbone):
    bb_names = {1: "alpha", 2: "beta", 3: "gamma", 5: "epsilon", 6: "zeta"}
    for tor in sorted(bb_names):
        n_bins = 3 if tor in (1, 3) else 2
        vals = ", ".join(f"{backbone[tor][b]:9.4f}" for b in range(1, n_bins + 1))
        label = bb_names[tor] + ":"
        out.write(f"    {label:10s} [{vals}]\n")


def _emit_sugar(out, sugar, bases):
    for slot, (name, _) in enumerate(SUGAR_TORSIONS):
        out.write(f"    {name}:\n")
        if name == "chi":
            for bi, base in enumerate(bases):
                vals = ", ".join(f"{sugar[bi, p, slot]:9.4f}" for p in range(N_PUCKER))
                out.write(f"      {base}: [{vals}]\n")
        else:
            vals = ", ".join(f"{sugar[0, p, slot]:9.4f}" for p in range(N_PUCKER))
            out.write(f"      all: [{vals}]\n")


def _emit_wells(out, wells, bases):
    fmt = lambda row: ", ".join(f"{v:7.4f}" for v in row)  # noqa: E731
    out.write(f"    pucker: [{fmt(wells['pucker'][0])}]\n")
    out.write("    alpha_gamma:  # rows alpha g+/t/g-, columns gamma g+/t/g-\n")
    for row in wells["alpha_gamma"][0]:
        out.write(f"      - [{fmt(row)}]\n")
    out.write("    bibii_given_pucker:  # rows BI/BII, columns north/south\n")
    for row in wells["bibii_given_pucker"][0]:
        out.write(f"      - [{fmt(row)}]\n")
    out.write("    alphanext_given_bibii:  # rows alpha(i+1) g+/t/g-, columns BI/BII\n")
    for row in wells["alphanext_given_bibii"][0]:
        out.write(f"      - [{fmt(row)}]\n")
    out.write("    chi_syn_given_pucker:  # anti/syn, by pucker, per base\n")
    for si, state in enumerate(("anti", "syn")):
        out.write(f"      {state}:\n")
        for bi, base in enumerate(bases):
            row = wells["chi_syn_given_pucker"][0][si, :, bi]
            out.write(f"        {base}: [{fmt(row)}]\n")


def emit_yaml(path, tables, provenance):
    """One file, each table split into its dna and rna parameter sets."""
    sections = (
        ("global_parameters", None),
        (
            "backbone_means",
            "mean backbone torsion by bin; alpha/gamma g+/t/g-, others BI/BII",
        ),
        ("sugar_means", "mean sugar torsion by pucker; only chi is base-dependent"),
        (
            "well_energies",
            "bin-population energies, -ln P shifted so the deepest well is 0.\n"
            "# Each bin assignment is charged once: the sugar torsions all read\n"
            "# pucker, and beta reads the previous residue's BI/BII.",
        ),
    )
    with open(path, "w") as out:
        for line in provenance:
            out.write(f"# {line}\n")
        for name, comment in sections:
            out.write(f"\n{'# ' + comment if comment else ''}\n" if comment else "\n")
            out.write(f"{name}:\n")
            for poly in POLYMERS:
                t = tables[poly]
                out.write(f"  {poly}:\n")
                if name == "global_parameters":
                    _emit_globals(out, t["sdev_obs"])
                elif name == "backbone_means":
                    _emit_backbone(out, t["backbone"])
                elif name == "sugar_means":
                    _emit_sugar(out, t["sugar"], BASE_ORDER[poly])
                else:
                    _emit_wells(out, t["wells"], BASE_ORDER[poly])


# ------------------------------------------------------------------ validate


def read_rosetta_stats(path):
    """Rosetta's own observations, for cross-checking the pipeline."""
    rows = []
    for line in open(path):
        f = line.split()
        if not f or f[0] != "DNA_DIHEDRALS":
            continue
        v = [float(x) for x in f[10:20]]
        rows.append(
            dict(
                base=f[2],
                pucker=int(f[5]),
                lower=f[6] == "--",
                upper=f[8] == "--",
                tor=dict(
                    alpha=v[0], beta=v[1], gamma=v[2], delta=v[3],
                    epsilon=v[4], zeta=v[5], chi=v[6],
                ),  # fmt: skip
            )
        )
    return rows


def validate(obs, stats_path):
    """Compare recomputed pucker and torsions against Rosetta's observations.

    Matches rows to nucleotides by torsion fingerprint, since Rosetta's residue
    index cannot be reconstructed.
    """
    rows = read_rosetta_stats(stats_path)
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
    by_fp = defaultdict(list)
    for o in obs:
        by_fp[round(o["tor"]["delta"], 1)].append(o)

    matched = pucker_agree = 0
    interior = [r for r in rows if not r["lower"] and not r["upper"]]
    for row in interior:
        for o in by_fp.get(round(row["tor"]["delta"], 1), ()):
            if all(
                o["tor"][n] is not None
                and abs(subtract_degree_angles(o["tor"][n], row["tor"][n])) < 0.25
                for n in names
            ):
                matched += 1
                pucker_agree += o["pucker"] == row["pucker"]
                break
    print(f"  Rosetta interior rows      {len(interior)}")
    print(f"  matched in our set         {matched}")
    if matched:
        print(f"  pucker index agreement     {pucker_agree}/{matched}")


# ----------------------------------------------------------------------- cli


def build_tables(obs, poly, verbose=False):
    """Every table for one polymer, from that polymer's observations."""
    bases = BASE_ORDER[poly]
    sugar, sugar_n = sugar_means(obs, bases)
    backbone, _ = backbone_means(obs)
    sparse = [
        (bases[b], pk, SUGAR_TORSIONS[s][0])
        for b in range(4)
        for pk in range(N_PUCKER)
        for s in range(N_SUGAR)
        if sugar_n[b, pk, s] < MIN_TORSIONS
    ]
    print(f"  {poly}: sugar bins below {MIN_TORSIONS} observations: {len(sparse)}")
    if sparse and verbose:
        for entry in sparse:
            print("     ", entry)
    return dict(
        sugar=sugar,
        backbone=backbone,
        wells=well_tables(obs, bases),
        sdev_obs=observed_sdev(obs, sugar, bases),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--structures", default=DNA_STRUCTURE_LIST)
    p.add_argument("--rna-structures", default=RNA_STRUCTURE_LIST)
    p.add_argument("--pdb-dir", default="na_structures")
    p.add_argument("--out")
    p.add_argument("--requery", action="store_true", help="rebuild the structure list")
    p.add_argument("--resolution", type=float, default=2.3)
    p.add_argument("--seqid", type=int, default=30)
    p.add_argument("--max-b", type=float, default=None, help="per-atom B cutoff")
    p.add_argument("--validate", metavar="BOUND_DNA_DIHEDRALS_TXT")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.requery:
        codes = query_rcsb(args.resolution, args.seqid)
        today = datetime.date.today().isoformat()
        with open(args.structures, "w") as f:
            f.write(
                f"# Protein-DNA X-ray structures <= {args.resolution} A, one"
                f" representative per {args.seqid}% seq-id cluster (protein entity).\n"
                f"# RCSB query {today}; regenerate with"
                f" na_torsion_param_import.py --requery\n"
            )
            f.write("\n".join(codes) + "\n")
        print(f"wrote {len(codes)} codes to {args.structures}")
        return

    def read_list(path):
        return [x.strip() for x in open(path) if x.strip() and not x.startswith("#")]

    # each list is swept for both polymers; a protein-DNA entry may carry an RNA
    # chain and vice versa, and the sugar decides which set a nucleotide joins
    obs, n_struct, n_skipped = [], 0, 0
    for path in (args.structures, args.rna_structures):
        o, ns, nk = observations(
            read_list(path), args.pdb_dir, args.max_b, args.verbose
        )
        obs += o
        n_struct += ns
        n_skipped += nk
    print(f"structures {n_struct} used, {n_skipped} skipped")

    if args.validate:
        validate([o for o in obs if o["poly"] == "dna"], args.validate)

    tables = {}
    counts = {}
    for poly in POLYMERS:
        sub = [o for o in obs if o["poly"] == poly]
        interior = [o for o in sub if not o["lower"] and not o["upper"]]
        counts[poly] = (len(sub), len(interior))
        print(f"  {poly}: nucleotides {len(sub)} ({len(interior)} interior)")
        tables[poly] = build_tables(sub, poly, args.verbose)

    if args.out:
        provenance = [
            "Nucleic acid torsion mean angles and well depths. Generated by",
            "tmol/support/scoring/na_torsion_param_import.py -- do not hand-edit.",
            f"DNA: {os.path.basename(args.structures)}, protein-DNA X-ray"
            f" <= {args.resolution} A, one per {args.seqid}% seq-id cluster;",
            f"     {counts['dna'][1]} interior nucleotides of {counts['dna'][0]}.",
            f"RNA: {os.path.basename(args.rna_structures)};",
            f"     {counts['rna'][1]} interior nucleotides of {counts['rna'][0]}.",
        ]
        emit_yaml(args.out, tables, provenance)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
