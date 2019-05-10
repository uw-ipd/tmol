import gzip
import numpy
import torch
import zarr
import os

rotamer_aliases = {
    "pro": numpy.array([[1, 3, 1, 1, 1, 1], [3, 1, 3, 2, 1, 1]], dtype=int)
}


def write_rotameric_data_for_aa(aa_lines, nchi, zgroup, rotamer_alias=None):
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

        probabilities[rot_ind, phi_ind, psi_ind] = float(cols[8])
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
        count_rots_in_ppbin[phi_ind, psi_ind] = count_for_ppbin + 1

    numpy.testing.assert_array_equal(
        count_rots_in_ppbin, nrots * numpy.ones([36, 36], dtype=int)
    )

    rotamer_group = zgroup.create_group("rotameric_data")
    rotamer_group.array("rotamers", sorted_rots_array)
    rotamer_group.array("probabilities", probabilities)
    rotamer_group.array("means", means)
    rotamer_group.array("stdvs", stdvs)
    rotamer_group.array("prob_sorted_rot_inds", prob_sorted_rots_by_phi_psi)
    rotamer_group.array("backbone_dihedral_start", [-180, -180])
    rotamer_group.array("backbone_dihedral_step", [10, 10])
    rotamer_group.array("rotamer_alias", rotamer_alias)


# def write_semi_rotameric_chi_table(
#    indent, rotchi_ind, lines_for_rotchi, n_rotameric_chi, trailing_comma, out_lines
# ):
#    prob_start_col = 5 + 3 * n_rotameric_chi
#    chi_prob_table = [[None for _a in range(36)] for _b in range(36)]
#    for line in lines_for_rotchi:
#        cols = line.split()
#        phi_ind, psi_ind = (int(dihe) // 10 + 18 for dihe in cols[1:3])
#        if phi_ind == 36 or psi_ind == 36:
#            continue
#        chi_prob_table[phi_ind][psi_ind] = cols[prob_start_col:]
#    out_lines.append(
#        "".join(
#            [
#                " " * indent,
#                '{"rotameric_rot_index": %d, "probabilities":\n' % rotchi_ind,
#            ]
#        )
#    )
#    write_3d_table(indent + 1, chi_prob_table, False, out_lines)
#    out_lines.append("".join([" " * indent, "}", "," if trailing_comma else ""]))


# def write_semi_rotameric_rotamer_definition(
#    indent, line, nchi, trailing_comma, output_lines
# ):
#    cols = line.split()
#    output_lines.append(
#        "".join(
#            [
#                " " * indent,
#                '{"rotamer": [',
#                ", ".join(cols[0:nchi]),
#                '], "left": ',
#                cols[6],
#                ', "right": ',
#                cols[8],
#                "}",
#                "," if trailing_comma else "",
#                "\n",
#            ]
#        )
#    )


def strip_comments(lines):
    return [line for line in lines if (len(line) > 0 and line[0] != "#")]


def write_rotameric_aa_dunbrack_library(aa3, aa_lines, nchi, zgroup, rotamer_alias):
    lib_group = zgroup.create_group(aa3)
    write_rotameric_data_for_aa(aa_lines, nchi, lib_group, rotamer_alias)


def write_semi_rotameric_aa_dunbrack_library(
    aa3,
    nchi,
    bb_rotamer_lines,
    bbdep_density_lines,
    ref_bbdep_density_lines,
    bbind_rotamer_def_lines,
    zgroup,
):
    print("Creating", aa3, "group")
    semirot_lib_group = zgroup.create_group(aa3)
    print("Created", aa3, "group")
    n_rotameric_chi = nchi - 1
    bb_rotamer_lines = strip_comments(bb_rotamer_lines)

    write_rotameric_data_for_aa(bb_rotamer_lines, nchi, semirot_lib_group)
    rotameric_data_group = semirot_lib_group["rotameric_data"]
    rotameric_rotamers = rotameric_data_group["rotamers"][:]
    n_rotamers = rotameric_rotamers.shape[0]
    allchi_rot_to_rot_ind = {
        tuple(rotameric_rotamers[i, :]): i for i in range(n_rotamers)
    }

    # read the chi labels, the number of non-rotameric chi samples, and the
    # non-rotameric chi start, step and period from the comments at the top
    # of the file, in the second to last line of comments
    for i, line in enumerate(ref_bbdep_density_lines):
        if len(line) == 0 or line[0] == "#":
            continue
        assert i >= 2
        desc_line = ref_bbdep_density_lines[i - 2]
        print("desc line:", desc_line)
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
    semirot_lib_group.attrs.update(
        nonrot_chi_start=nonrot_chi_start,
        nonrot_chi_step=nonrot_chi_step,
        nonrot_chi_period=nonrot_chi_period,
    )

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

    semirot_lib_group.array("rotameric_chi_rotamers", sorted_rotchi_rotamers_array)

    # non-rotameric-chi (nrc) probabilities
    nrc_probs = numpy.zeros([n_rotchi_rotamers, 36, 36, nrc_n_samples], dtype=float)
    for line in bbdep_density_lines:
        if len(line) == 0 or line[0] == "#":
            continue
        cols = line.split()
        phi_ind, psi_ind = (int(x) // 10 + 18 for x in cols[1:3])
        if phi_ind >= 36 or psi_ind >= 36:
            continue
        rot = tuple(int(x) for x in cols[4 : (4 + n_rotameric_chi)])
        chi = [float(x) for x in cols[(5 + 3 * n_rotameric_chi) :]]
        rot_ind = rotameric_rot_to_rot_ind[rot]
        nrc_probs[rot_ind, phi_ind, psi_ind, :] = chi
    semirot_lib_group.array("nonrotameric_chi_probabilities", nrc_probs)

    rotamer_boundaries = numpy.zeros([n_rotamers, 2], dtype=float)
    count_rotamers = 0
    for i, line in enumerate(bbind_rotamer_def_lines):
        if len(line) == 0 or line[0] == "#":
            continue
        cols = line.split()
        rot = tuple(int(x) for x in cols[0:nchi])
        rot_ind = allchi_rot_to_rot_ind[rot]
        # nab columns 6 and 8
        rotamer_boundaries[rot_ind, :] = [float(x) for x in cols[6:9:2]]

    semirot_lib_group.array("rotamer_boundaries", rotamer_boundaries)


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

    store = zarr.ZipStore(out_path)
    zgroup = zarr.group(store=store)

    # Rotameric residues:
    # CYS, ILE, LYS, LEU, MET, PRO, ARG, SER, THR, VAL
    rotameric_aas = ["cys", "ile", "lys", "met", "pro", "arg", "ser", "thr", "val"]
    rotameric_zgroup = zgroup.create_group("rotameric_tables")
    rotameric_zgroup.attrs.update(tables=rotameric_aas)
    lib_files = [
        os.path.join(path_to_db_dir, x + ".bbdep.rotamers.lib.gz")
        for x in rotameric_aas
    ]
    for i, lib_file in enumerate(lib_files):
        print("processing", lib_file)
        with gzip.GzipFile(lib_file) as fid:
            lines = [x.decode("utf-8") for x in fid.readlines()]
        lines = strip_comments(lines)
        aa = rotameric_aas[i]
        rotamer_alias = None
        if aa in rotamer_aliases:
            rotamer_alias = rotamer_aliases[aa]
        write_rotameric_aa_dunbrack_library(
            rotameric_aas[i],
            lines,
            nchi_for_aa[aa.upper()],
            rotameric_zgroup,
            rotamer_alias,
        )

    semirot_aas = ["asp", "glu", "phe", "his", "asn", "gln", "trp", "tyr"]
    semirotameric_zgroup = zgroup.create_group("semirotameric_tables")
    semirotameric_zgroup.attrs.update(tables=semirot_aas)
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
    for i, files in enumerate(lib_files):
        print("opening files", files)
        with gzip.GzipFile(files[0]) as fid:
            rotamer_lines = [x.decode("utf-8") for x in fid.readlines()]
        with gzip.GzipFile(files[1]) as fid:
            density_lines = [x.decode("utf-8") for x in fid.readlines()]
        with gzip.GzipFile(files[2]) as fid:
            ref_density_lines = [x.decode("utf-8") for x in fid.readlines()]
        with gzip.GzipFile(files[3]) as fid:
            bbind_rotamer_lines = [x.decode("utf-8") for x in fid.readlines()]
        print("processing", files[0], files[1], files[2], files[3])
        write_semi_rotameric_aa_dunbrack_library(
            semirot_aas[i],
            nchi_for_aa[semirot_aas[i].upper()],
            rotamer_lines,
            density_lines,
            ref_density_lines,
            bbind_rotamer_lines,
            semirotameric_zgroup,
        )

    store.close()


if __name__ == "__main__":

    write_binary_version_of_dunbrack_rotamer_library(
        "/home/andrew/rosetta/GIT/Rosetta/main/database/rotamer/beta_nov2016/",
        "/home/andrew/rosetta/GIT/Rosetta/main/database/rotamer/ExtendedOpt1-5/",
        "dunbrack.bin",
    )
    # open("dunbrack_library2.json", "w").writelines(outlines)
