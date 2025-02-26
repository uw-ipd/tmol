import gzip
import numpy
import zarr
import os
import torch
import yaml
import attr
import cattr
from tmol.database.scoring.dunbrack_libraries import (
    DunbrackRotamerLibrary,
    RotamericDataForAA,
    RotamericAADunbrackLibrary,
    SemiRotamericAADunbrackLibrary,
)

rotamer_aliases = {
    "pro": numpy.array([[1, 3, 1, 1, 1, 1], [3, 1, 3, 2, 1, 1]], dtype=int)
}


def write_rotameric_data_for_aa(aa_lines, nchi, rotamer_alias=None):
    print(nchi)
    rotamers = []
    rotamers_set = set([])
    for line in aa_lines:
        cols = line.split()
        rot_id = tuple([int(x) for x in cols[4 : (4 + nchi)]])
        if rot_id not in rotamers_set:
            rotamers.append(rot_id)
            rotamers_set.add(rot_id)

    sorted_rots = sorted(rotamers)
    sorted_rots_array = numpy.array(sorted_rots, dtype=int)
    # sorted_rots_str = [[str(well) for well in rot] for rot in sorted_rots]
    rot_to_rot_ind = {x: i for i, x in enumerate(sorted_rots)}
    nrots = len(sorted_rots)

    probabilities = numpy.zeros([nrots, 36, 36], dtype=float)
    means = numpy.zeros([nrots, 36, 36, nchi], dtype=float)
    stdvs = numpy.zeros([nrots, 36, 36, nchi], dtype=float)
    prob_sorted_rots_by_phi_psi = numpy.zeros([36, 36, nrots], dtype=int)
    rotprob_by_phi_psi_order = numpy.ones([36, 36, nrots], dtype=float)
    count_rots_in_ppbin = numpy.zeros([36, 36], dtype=int)
    if rotamer_alias is None:
        rotamer_alias = numpy.zeros([0, nchi], dtype=int)

    for line in aa_lines:
        cols = line.split()
        phi_ind, psi_ind = (int(dihe) // 10 + 18 for dihe in cols[1:3])
        if phi_ind >= 36 or psi_ind >= 36:
            continue
        rot_id = tuple([int(x) for x in cols[4 : (4 + nchi)]])
        rot_ind = rot_to_rot_ind[rot_id]

        prob = float(cols[8])
        probabilities[rot_ind, phi_ind, psi_ind] = prob
        means[rot_ind, phi_ind, psi_ind, :] = numpy.array(
            [float(x) for x in cols[9 : (9 + nchi)]], dtype=float
        )
        stdvs[rot_ind, phi_ind, psi_ind, :] = numpy.array(
            [float(x) for x in cols[13 : (13 + nchi)]], dtype=float
        )
        count_for_ppbin = count_rots_in_ppbin[phi_ind, psi_ind]
        prob_sorted_rots_by_phi_psi[phi_ind, psi_ind, count_for_ppbin] = rot_to_rot_ind[
            rot_id
        ]
        rotprob_by_phi_psi_order[phi_ind, psi_ind, count_for_ppbin] = -1 * prob
        count_rots_in_ppbin[phi_ind, psi_ind] = count_for_ppbin + 1

    rotprob_by_phi_psi_order_inds = numpy.argsort(rotprob_by_phi_psi_order, axis=2)
    prob_sorted_rots_by_phi_psi = numpy.take_along_axis(
        prob_sorted_rots_by_phi_psi, rotprob_by_phi_psi_order_inds, axis=2
    )

    numpy.testing.assert_array_equal(
        count_rots_in_ppbin, nrots * numpy.ones([36, 36], dtype=int)
    )

    return sorted_rots_array, RotamericDataForAA(
        rotamers=torch.tensor(sorted_rots_array),
        rotamer_probabilities=torch.tensor(probabilities, dtype=torch.float32),
        rotamer_means=torch.tensor(means, dtype=torch.float32),
        rotamer_stdvs=torch.tensor(stdvs, dtype=torch.float32),
        prob_sorted_rot_inds=torch.tensor(prob_sorted_rots_by_phi_psi),
        backbone_dihedral_start=torch.tensor([-180, -180], dtype=torch.float32),
        backbone_dihedral_step=torch.tensor([10, 10], dtype=torch.float32),
        rotamer_alias=torch.tensor(rotamer_alias),
    )


def strip_comments(lines):
    return [line for line in lines if (len(line) > 0 and line[0] != "#")]


def write_rotameric_aa_dunbrack_library(aa3, lines, nchi_for_aa, rotamer_alias):
    _, data_for_aa = write_rotameric_data_for_aa(lines, nchi_for_aa, rotamer_alias)
    print(type(data_for_aa.rotamers))
    return RotamericAADunbrackLibrary(aa3, data_for_aa)


def write_semi_rotameric_aa_dunbrack_library(
    aa3,
    nchi,
    bb_rotamer_lines,
    bbdep_density_lines,
    ref_bbdep_density_lines,
    bbind_rotamer_def_lines,
):
    # print("Creating", aa3, "group")
    # print("Created", aa3, "group")
    n_rotameric_chi = nchi - 1
    bb_rotamer_lines = strip_comments(bb_rotamer_lines)

    rots, rotameric_data = write_rotameric_data_for_aa(bb_rotamer_lines, nchi)
    n_rotamers = rots.shape[0]
    allchi_rot_to_rot_ind = {tuple(rots[i, :]): i for i in range(n_rotamers)}

    # read the chi labels, the number of non-rotameric chi samples, and the
    # non-rotameric chi start, step and period from the comments at the top
    # of the file, in the second to last line of comments
    for i, line in enumerate(ref_bbdep_density_lines):
        if len(line) == 0 or line[0] == "#":
            continue
        assert i >= 2
        desc_line = ref_bbdep_density_lines[i - 2]
        # print("desc line:", desc_line)
        cols = desc_line[1:].split()
        chi_labels = [int(x) for x in cols[(5 + 3 * n_rotameric_chi) :]]
        nrc_n_samples = len(chi_labels)
        nonrot_chi_start = chi_labels[0]
        nonrot_chi_step = chi_labels[1] - nonrot_chi_start
        nonrot_chi_period = chi_labels[-1] - nonrot_chi_start + nonrot_chi_step
        break

    # nonrot_chi_sampling_data = numpy.zeros((3,), dtype=float)
    # nonrot_chi_sampling_data[0] = nonrot_chi_start
    # nonrot_chi_sampling_data[1] = nonrot_chi_step
    # nonrot_chi_sampling_data[2] = nonrot_chi_period
    # semirot_lib_group.array("nonrot_chi_sampling_data", nonrot_chi_sampling_data)
    # semirot_lib_group.attrs.update(
    # nonrot_chi_start=nonrot_chi_start,
    # nonrot_chi_step=nonrot_chi_step,
    # nonrot_chi_period=nonrot_chi_period,
    # )

    rotchi_rotamers = []
    rotchi_rotamers_set = set([])
    rot_cols = slice(4, 4 + n_rotameric_chi)
    for line in bbdep_density_lines:
        if len(line) == 0 or line[0] == "#":
            continue
        cols = line.split()
        rot = tuple(int(x) for x in cols[rot_cols])
        if rot not in rotchi_rotamers_set:
            rotchi_rotamers.append(rot)
            rotchi_rotamers_set.add(rot)

    sorted_rotchi_rotamers = sorted(rotchi_rotamers)
    sorted_rotchi_rotamers_array = numpy.array(sorted_rotchi_rotamers, dtype=int)
    n_rotchi_rotamers = len(sorted_rotchi_rotamers)
    rotameric_rot_to_rot_ind = {x: i for i, x in enumerate(sorted_rotchi_rotamers)}

    # semirot_lib_group.array("rotameric_chi_rotamers", sorted_rotchi_rotamers_array)

    # non-rotameric-chi (nrc) probabilities
    nrc_probs = numpy.zeros([n_rotchi_rotamers, 36, 36, nrc_n_samples], dtype=float)
    for line in bbdep_density_lines:
        if len(line) == 0 or line[0] == "#":
            continue
        cols = line.split()
        phi_ind, psi_ind = (int(x) // 10 + 18 for x in cols[1:3])
        if phi_ind >= 36 or psi_ind >= 36:
            continue
        base_prob = float(cols[4 + n_rotameric_chi])
        rot = tuple(int(x) for x in cols[4 : (4 + n_rotameric_chi)])
        chi = [base_prob * float(x) for x in cols[(5 + 3 * n_rotameric_chi) :]]
        rot_ind = rotameric_rot_to_rot_ind[rot]
        nrc_probs[rot_ind, phi_ind, psi_ind, :] = chi
    # semirot_lib_group.array("nonrotameric_chi_probabilities", nrc_probs)

    rotamer_boundaries = numpy.zeros([n_rotamers, 2], dtype=float)
    for line in bbind_rotamer_def_lines:
        if len(line) == 0 or line[0] == "#":
            continue
        cols = line.split()
        rot = tuple(int(x) for x in cols[0:nchi])
        rot_ind = allchi_rot_to_rot_ind[rot]
        # nab columns 6 and 8
        rotamer_boundaries[rot_ind, :] = [float(x) for x in cols[6:9:2]]

    # semirot_lib_group.array("rotamer_boundaries", rotamer_boundaries)

    return SemiRotamericAADunbrackLibrary(
        table_name=aa3,
        rotameric_data=rotameric_data,
        non_rot_chi_start=nonrot_chi_start,
        non_rot_chi_step=nonrot_chi_step,
        non_rot_chi_period=nonrot_chi_period,
        rotameric_chi_rotamers=torch.tensor(sorted_rotchi_rotamers_array),
        nonrotameric_chi_probabilities=torch.tensor(nrc_probs, dtype=torch.float32),
        rotamer_boundaries=torch.tensor(rotamer_boundaries, dtype=torch.float32),
    )


def write_binary_version_of_dunbrack_rotamer_library(
    path_to_db_dir, path_to_reference_db_dir, out_path
):
    nchi_for_aa = {
        "CYS": 1,
        "ASP": 2,
        "GLU": 3,
        "PHE": 2,
        "HIS": 2,
        "ILE": 2,
        "LYS": 4,
        "LEU": 2,
        "MET": 3,
        "ASN": 2,
        "PRO": 3,
        "GLN": 3,
        "ARG": 4,
        "SER": 1,
        "THR": 1,
        "VAL": 1,
        "TRP": 2,
        "TYR": 2,
    }

    path_lookup = "tmol/database/default/scoring/dunbrack.yaml"

    dun_lookup = None
    with open(path_lookup, "r") as infile_lookup:
        raw = yaml.load(infile_lookup, Loader=yaml.FullLoader)
        dun_lookup = cattr.structure(
            raw["dunbrack_lookup"], attr.fields(DunbrackRotamerLibrary).dun_lookup.type
        )

    # Rotameric residues:
    # CYS, ILE, LYS, LEU, MET, PRO, ARG, SER, THR, VAL
    rotameric_aas = [
        "cys",
        "ile",
        "lys",
        "leu",
        "met",
        "pro",
        "arg",
        "ser",
        "thr",
        "val",
    ]
    lib_files = [
        os.path.join(path_to_db_dir, x + ".bbdep.rotamers.lib.gz")
        for x in rotameric_aas
    ]
    rdls = []
    for i, lib_file in enumerate(lib_files):
        # print("processing", lib_file)
        with gzip.GzipFile(lib_file) as fid:
            lines = [x.decode("utf-8") for x in fid.readlines()]
        lines = strip_comments(lines)
        aa = rotameric_aas[i]
        rotamer_alias = None
        if aa in rotamer_aliases:
            rotamer_alias = rotamer_aliases[aa]
        rot_lib = write_rotameric_aa_dunbrack_library(
            rotameric_aas[i], lines, nchi_for_aa[aa.upper()], rotamer_alias
        )
        rdls.append(rot_lib)

    semirot_aas = ["asp", "glu", "phe", "his", "asn", "gln", "trp", "tyr"]
    ##semirotameric_zgroup = zgroup.create_group("semirotameric_tables")
    # semirotameric_zgroup.attrs.update(tables=semirot_aas)
    lib_files = [
        (
            os.path.join(path_to_db_dir, x + ".bbdep.rotamers.lib.gz"),
            os.path.join(path_to_db_dir, x + ".bbdep.densities.lib.gz"),
            os.path.join(path_to_reference_db_dir, x + ".bbdep.densities.lib.gz"),
            os.path.join(
                path_to_db_dir,
                x + ".bbind.chi" + str(nchi_for_aa[x.upper()]) + ".Definitions.lib.gz",
            ),
        )
        for x in semirot_aas
    ]
    srdls = []
    for i, files in enumerate(lib_files):
        # print("opening files", files)
        with gzip.GzipFile(files[0]) as fid:
            rotamer_lines = [x.decode("utf-8") for x in fid.readlines()]
        with gzip.GzipFile(files[1]) as fid:
            density_lines = [x.decode("utf-8") for x in fid.readlines()]
        with gzip.GzipFile(files[2]) as fid:
            ref_density_lines = [x.decode("utf-8") for x in fid.readlines()]
        with gzip.GzipFile(files[3]) as fid:
            bbind_rotamer_lines = [x.decode("utf-8") for x in fid.readlines()]
        # print("processing", files[0], files[1], files[2], files[3])
        srot_lib = write_semi_rotameric_aa_dunbrack_library(
            semirot_aas[i],
            nchi_for_aa[semirot_aas[i].upper()],
            rotamer_lines,
            density_lines,
            ref_density_lines,
            bbind_rotamer_lines,
        )
        srdls.append(srot_lib)

    return DunbrackRotamerLibrary(
        dun_lookup=dun_lookup,
        rotameric_libraries=tuple(rdls),
        semi_rotameric_libraries=tuple(srdls),
    )


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.save(
        write_binary_version_of_dunbrack_rotamer_library(
            "/home/jflat06/foldit/develop/database/rotamer/beta_nov2016/",
            "/home/jflat06/foldit/develop/database/rotamer/ExtendedOpt1-5/",
            "/home/jflat06/dunbrack.bin",
        ),
        "DUNTEST.tzip",
    )
    # open("dunbrack_library2.json", "w").writelines(outlines)
