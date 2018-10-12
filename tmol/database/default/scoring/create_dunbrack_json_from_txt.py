import gzip
import torch
import zarr


def write_2d_table(indent, table, trailing_comma, out_lines):
    for i, row in enumerate(table):
        nextline = [" " * indent, "[[" if i == 0 else " ["]
        nextline.append(", ".join(row))
        if i == len(table) - 1:
            nextline.append("]]")
            if trailing_comma:
                nextline.append(",")
            nextline.append("\n")
        else:
            nextline.append("],\n")
        out_lines.append("".join(nextline))


def write_3d_table(indent, table, trailing_comma, out_lines):
    for i, row in enumerate(table):
        nextline = [" " * indent, "[[" if i == 0 else " ["]
        for j, col in enumerate(row):
            nextline.append(
                "".join(
                    [(" " * (indent + 2) if j != 0 else ""), "[", ", ".join(col), "]"]
                )
            )

            if j == len(row) - 1:
                nextline.append("]")
                if i == len(table) - 1:
                    nextline.append("]")
                    if trailing_comma:
                        nextline.append(",")
                else:
                    nextline.append(",")
            else:
                nextline.append(",")

            nextline.append("\n")

            out_lines.append("".join(nextline))
            nextline = []


def write_dunbrack_rotamer_table(
    indent, rotamer_lines, nchi, index, trailing_comma, out_lines
):
    probabilities = [[""] * 36 for _ in range(36)]
    means = [[[""] * nchi for i in range(36)] for j in range(36)]
    stdevs = [[[""] * nchi for i in range(36)] for j in range(36)]
    for line in rotamer_lines:
        cols = line.split()
        phi_ind, psi_ind = (int(dihe) // 10 + 18 for dihe in cols[1:3])
        if phi_ind >= 36 or psi_ind >= 36:
            continue
        probabilities[phi_ind][psi_ind] = cols[8]
        means[phi_ind][psi_ind] = cols[9 : (9 + nchi)]
        stdevs[phi_ind][psi_ind] = cols[13 : (13 + nchi)]
    out_lines.append(
        "".join(
            [" " * indent, '{"rotamer_index": ', str(index), ', "probabilities":\n']
        )
    )
    write_2d_table(indent + 1, probabilities, True, out_lines)
    out_lines.append("".join([" " * indent, '"means":\n']))

    write_3d_table(indent + 1, means, True, out_lines)
    out_lines.append("".join([" " * indent, '"stdevs":\n']))
    write_3d_table(indent + 1, stdevs, False, out_lines)
    out_lines.append("".join([" " * indent, "}", "," if trailing_comma else "", "\n"]))


def write_rotameric_data_for_aa(indent, aa_lines, nchi, trailing_comma, out_lines):
    rotamers = []
    lines_for_rotamers = {}
    for line in aa_lines:
        cols = line.split()
        rot_id = tuple([int(x) for x in cols[4 : (4 + nchi)]])
        if rot_id not in lines_for_rotamers:
            rotamers.append(rot_id)
            lines_for_rotamers[rot_id] = []
        lines_for_rotamers[rot_id].append(line)
    sorted_rots = sorted(rotamers)
    sorted_rots_str = [[str(well) for well in rot] for rot in sorted_rots]
    rot_to_rot_ind = {x: i for i, x in enumerate(sorted_rots)}

    # create a 3d table of rotamer inds
    # sorted by probability -- the original dunbrack
    # table lists the rotamers in decreasing order by
    # probability; map these rotamers to their index
    prob_sorted_rot_by_phi_psi = [[list() for _a in range(36)] for _b in range(36)]
    for line in aa_lines:
        cols = line.split()
        phi_ind, psi_ind = (int(dihe) // 10 + 18 for dihe in cols[1:3])
        # print("phi", cols[1], phi_ind, "psi", cols[2], psi_ind)
        if phi_ind >= 36 or psi_ind >= 36:
            continue
        rot_id = tuple([int(x) for x in cols[4 : (4 + nchi)]])
        prob_sorted_rot_by_phi_psi[phi_ind][psi_ind].append(str(rot_to_rot_ind[rot_id]))

    out_lines.append(
        "".join(
            [" " * indent, '{"nchi": %d,\n' % nchi, " " * (indent + 1), '"rotamers":\n']
        )
    )
    write_2d_table(indent + 2, sorted_rots_str, True, out_lines)

    out_lines.append("".join([" " * (indent + 1), '"rotamer_tables": [\n']))
    for i, rot in enumerate(sorted_rots):
        write_dunbrack_rotamer_table(
            indent + 2,
            lines_for_rotamers[rot],
            nchi,
            i,
            i != len(sorted_rots) - 1,
            out_lines,
        )
    out_lines.append("".join([" " * (indent + 1), "],\n"]))
    out_lines.append("".join([" " * (indent + 1), '"prob_sorted_rot_ind":\n']))
    write_3d_table(indent + 2, prob_sorted_rot_by_phi_psi, True, out_lines)
    out_lines.append(
        "".join(
            [
                " " * (indent + 1),
                '"backbone_dihedral_start": ',
                "[-180, -180],\n",
                " " * (indent + 1),
                '"backbone_dihedral_step": ',
                "[10, 10]\n",
                " " * indent,
                "}",
                "," if trailing_comma else "",
                "\n",
            ]
        )
    )


def write_semi_rotameric_chi_table(
    indent, rotchi_ind, lines_for_rotchi, n_rotameric_chi, trailing_comma, out_lines
):
    prob_start_col = 5 + 3 * n_rotameric_chi
    chi_prob_table = [[None for _a in range(36)] for _b in range(36)]
    for line in lines_for_rotchi:
        cols = line.split()
        phi_ind, psi_ind = (int(dihe) // 10 + 18 for dihe in cols[1:3])
        if phi_ind == 36 or psi_ind == 36:
            continue
        chi_prob_table[phi_ind][psi_ind] = cols[prob_start_col:]
    out_lines.append(
        "".join(
            [
                " " * indent,
                '{"rotameric_rot_index": %d, "probabilities":\n' % rotchi_ind,
            ]
        )
    )
    write_3d_table(indent + 1, chi_prob_table, False, out_lines)
    out_lines.append("".join([" " * indent, "}", "," if trailing_comma else ""]))


def write_semi_rotameric_rotamer_definition(
    indent, line, nchi, trailing_comma, output_lines
):
    cols = line.split()
    output_lines.append(
        "".join(
            [
                " " * indent,
                '{"rotamer": [',
                ", ".join(cols[0:nchi]),
                '], "left": ',
                cols[6],
                ', "right": ',
                cols[8],
                "}",
                "," if trailing_comma else "",
                "\n",
            ]
        )
    )


def strip_comments(lines):
    return [line for line in lines if (len(line) > 0 and line[0] != "#")]


def write_rotameric_aa_dunbrack_library(
    indent, aa3, aa_lines, nchi, trailing_comma, out_lines
):
    out_lines.append(
        "".join([" " * indent, '{"table_name": "%s", "rotameric_data":\n' % aa3])
    )
    aa_lines = strip_comments(aa_lines)
    write_rotameric_data_for_aa(indent + 1, aa_lines, nchi, False, out_lines)
    out_lines.append("".join([" " * indent, "}", "," if trailing_comma else "", "\n"]))


def write_semi_rotameric_aa_dunbrack_library(
    indent,
    aa3,
    nchi,
    bb_rotamer_lines,
    bbdep_density_lines,
    bbind_rotamer_def_lines,
    trailing_comma,
    out_lines,
):
    bb_rotamer_lines = strip_comments(bb_rotamer_lines)
    # bbdep_density_lines = strip_comments(bbdep_density_lines)
    # bbind_rotamer_def_lines = strip_comments(bbind_rotamer_def_lines)

    n_rotameric_chi = nchi - 1

    out_lines.append(
        "".join([" " * indent, '{"table_name": "%s", "rotameric_data":\n' % aa3])
    )
    write_rotameric_data_for_aa(indent + 2, bb_rotamer_lines, nchi, True, out_lines)

    # second to last line of comments will have the start, step, and periodicity
    for i, line in enumerate(bbdep_density_lines):
        if len(line) == 0 or line[0] == "#":
            continue
        assert i >= 2
        desc_line = bbdep_density_lines[i - 2]
        cols = desc_line[1:].split()
        chi_labels = [int(x) for x in cols[(5 + 3 * n_rotameric_chi) :]]
        nonrot_chi_start = chi_labels[0]
        nonrot_chi_step = chi_labels[1] - nonrot_chi_start
        nonrot_chi_period = chi_labels[-1] - nonrot_chi_start + nonrot_chi_step
        break

    out_lines.append(
        "".join([" " * (indent + 1), '"non_rot_chi_start": %f,\n' % nonrot_chi_start])
    )
    out_lines.append(
        "".join([" " * (indent + 1), '"non_rot_chi_step": %f,\n' % nonrot_chi_step])
    )
    out_lines.append(
        "".join(
            [" " * (indent + 1), '"non_rot_chi_periodicity": %f,\n' % nonrot_chi_period]
        )
    )

    rotchi_rotamers = []
    rotchi_rotamer_lines = {}
    rot_cols = slice(4, 4 + n_rotameric_chi)
    for line in bbdep_density_lines:
        if len(line) == 0 or line[0] == "#":
            continue
        cols = line.split()
        rot = tuple(int(x) for x in cols[rot_cols])
        if rot not in rotchi_rotamer_lines:
            rotchi_rotamer_lines[rot] = []
            rotchi_rotamers.append(rot)
        rotchi_rotamer_lines[rot].append(line)
    sorted_rotchi_rotamers = sorted(rotchi_rotamers)
    sorted_rotchi_rotamers_str = [
        [str(well) for well in rot] for rot in sorted_rotchi_rotamers
    ]
    rot_to_rot_ind = {x: i for i, x in enumerate(sorted_rotchi_rotamers)}

    out_lines.append("".join([" " * (indent + 1), '"rotameric_chi_rotamers":\n']))
    write_2d_table(indent + 2, sorted_rotchi_rotamers_str, True, out_lines)

    out_lines.append(
        "".join([" " * (indent + 1), '"nonrotameric_chi_probabilities": [\n'])
    )
    for i, rot in enumerate(sorted_rotchi_rotamers):
        write_semi_rotameric_chi_table(
            indent + 2,
            rot_to_rot_ind[rot],
            rotchi_rotamer_lines[rot],
            n_rotameric_chi,
            i != len(sorted_rotchi_rotamers) - 1,
            out_lines,
        )

    out_lines.append("".join([" " * (indent + 1), "],\n"]))

    out_lines.append("".join([" " * (indent + 1), '"rotamer_definitions": [\n']))
    bbind_rotamer_def_lines = strip_comments(bbind_rotamer_def_lines)
    for i, line in enumerate(bbind_rotamer_def_lines):
        write_semi_rotameric_rotamer_definition(
            indent + 2, line, nchi, i != len(bbind_rotamer_def_lines) - 1, out_lines
        )
    out_lines.append("".join([" " * (indent + 1), "]\n"]))
    out_lines.append(
        "".join([" " * (indent), "}", "," if trailing_comma else "", "\n"])
    )


# def write_dunbrack_library_to_json():
#    aa3 = None
#    for line in bbdep_rotamer_lines:
#        if len(line) == 0 or line[0] == "#":
#            continue
#        aa3 = line.split()[0]
#        break
#    nchi_for_aa = {
#        "CYS": 1,
#        "ASP": 2,
#        "GLU": 3,
#        "PHE": 2,
#        "HIS": 2,
#        "ILE": 2,
#        "LYS": 4,
#        "LEU": 2,
#        "MET": 3,
#        "ASN": 2,
#        "PRO": 3,
#        "GLN": 3,
#        "ARG": 4,
#        "SER": 1,
#        "THR": 1,
#        "VAL": 1,
#        "TRP": 2,
#        "TYR": 2,
#        }
#    nchi = nchi_for_aa[aa3]


def write_json_version_of_dunbrack_rotamer_library(path_to_db_dir):
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

    out_lines = []
    out_lines.append('{"rotameric_libraries": [\n')

    # Rotameric residues:
    # CYS, ILE, LYS, LEU, MET, PRO, ARG, SER, THR, VAL
    rotameric_aas = ["cys", "ile", "lys", "met", "pro", "arg", "ser", "thr", "val"]
    lib_files = [(path_to_db_dir + x + ".bbdep.rotamers.lib.gz") for x in rotameric_aas]
    first = True
    for i, lib_file in enumerate(lib_files):
        print("processing", lib_file)
        with gzip.GzipFile(lib_file) as fid:
            lines = [x.decode("utf-8") for x in fid.readlines()]
        aa = rotameric_aas[i]
        write_rotameric_aa_dunbrack_library(
            2,
            rotameric_aas[i],
            lines,
            nchi_for_aa[aa.upper()],
            i != len(lib_files) - 1,
            out_lines,
        )

    out_lines.append(" ],\n")
    out_lines.append(' "semi_rotameric_libraries": [\n')

    semirot_aas = ["asp", "glu", "phe", "his", "asn", "gln", "trp", "tyr"]
    lib_files = [
        (
            path_to_db_dir + x + ".bbdep.rotamers.lib.gz",
            path_to_db_dir + x + ".bbdep.densities.lib.gz",
            path_to_db_dir
            + x
            + ".bbind.chi"
            + str(nchi_for_aa[x.upper()])
            + ".Definitions.lib.gz",
        )
        for x in semirot_aas
    ]
    for i, files in enumerate(lib_files):
        with gzip.GzipFile(files[0]) as fid:
            rotamer_lines = [x.decode("utf-8") for x in fid.readlines()]
        with gzip.GzipFile(files[1]) as fid:
            density_lines = [x.decode("utf-8") for x in fid.readlines()]
        with gzip.GzipFile(files[2]) as fid:
            bbind_rotamer_lines = [x.decode("utf-8") for x in fid.readlines()]
        print("processing", files[0])
        write_semi_rotameric_aa_dunbrack_library(
            2,
            semirot_aas[i],
            nchi_for_aa[semirot_aas[i].upper()],
            rotamer_lines,
            density_lines,
            bbind_rotamer_lines,
            i != len(semirot_aas) - 1,
            out_lines,
        )
        first = False

    out_lines.append(" ]\n")
    out_lines.append("}\n")

    return out_lines


if __name__ == "__main__":
    # fname = "/Users/andrew/rosetta/GIT/Rosetta/main/database/rotamer/ExtendedOpt1-5/gln.bbdep.rotamers.lib.gz"
    # with gzip.GzipFile(fname,"r") as fid:
    #    decoded = [line.decode("utf-8") for line in fid.readlines()]
    #    rotamer_lines = [line for line in decoded if (len(line) > 0 and line[0] != "#")]
    # out_lines = []
    # write_dunbrack_rotamers_for_aa(0,rotamer_lines,3,False,out_lines)
    # with open("dummy_gln_rots2.json","w") as fid:
    #    fid.writelines(out_lines)

    outlines = write_json_version_of_dunbrack_rotamer_library(
        "/Users/andrew/rosetta/GIT/Rosetta/main/database/rotamer/ExtendedOpt1-5/"
    )
    open("dunbrack_library2.json", "w").writelines(outlines)
