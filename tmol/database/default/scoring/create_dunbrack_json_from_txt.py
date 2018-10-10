import gzip


# def write_dunbrack_rotamer_entry(line, first, nchi, indent, output_lines):
#     cols = line.split()
#
#     rotamer = tuple(cols[4 : (4 + nchi)])
#     prob = cols[8]
#     means = tuple(cols[9 : (9 + nchi)])
#     stdv = tuple(cols[13 : (13 + nchi)])
#     output_lines.append(
#         "".join(
#             [
#                 " " * indent,
#                 "" if first else ",",
#                 '{"rotamer": [',
#                 ", ".join(rotamer),
#                 '], "prob": ',
#                 prob,
#                 ', "means": [',
#                 ", ".join(means),
#                 '], "stdev": [',
#                 ", ".join(stdv),
#                 "]}\n",
#             ]
#         )
#     )
#
#
# def write_mainchain_bin_rotamer_entries(lines, first, nchi, indent, output_lines):
#     phi_psi_bin = tuple(lines[0].split()[1:3])
#     output_lines.append(
#         "".join(
#             [
#                 " " * indent,
#                 "" if first else ",",
#                 '{"bb_dihedrals": [',
#                 ", ".join(phi_psi_bin),
#                 '], "sorted_rotamers": [\n',
#             ]
#         )
#     )
#     first = True
#     for line in lines:
#         write_dunbrack_rotamer_entry(line, first, nchi, indent + 1, output_lines)
#         first = False
#     output_lines.append("".join([" " * indent, "]}\n"]))
#
#
# def write_aa_rotameric_dunbrack_library(lines, first, indent, output_lines):
#     nchi_for_aa = {
#         "CYS": 1,
#         "ASP": 1,
#         "GLU": 2,
#         "PHE": 2,
#         "HIS": 2,
#         "ILE": 2,
#         "LYS": 4,
#         "LEU": 2,
#         "MET": 3,
#         "ASN": 2,
#         "PRO": 3,
#         "GLN": 3,
#         "ARG": 4,
#         "SER": 1,
#         "THR": 1,
#         "VAL": 1,
#         "TRP": 2,
#         "TYR": 2,
#     }
#     for line in lines:
#         if len(line) == 0 or line[0] == "#":
#             continue
#         aa3 = line.split()[0]
#         pp = tuple(line.split()[1:3])
#         break
#     nchi = nchi_for_aa[aa3]
#     output_lines.append(
#         "".join(
#             [
#                 " " * indent,
#                 "" if first else ",",
#                 '{"aa_name": "',
#                 aa3,
#                 '", "mc_dihedral_entries": [\n',
#             ]
#         )
#     )
#     pp_lines = []
#     first = True
#     for line in lines:
#         if len(line) == 0 or line[0] == "#":
#             continue
#         next_pp = tuple(line.split()[1:3])
#         if pp != next_pp:
#             write_mainchain_bin_rotamer_entries(
#                 pp_lines, first, nchi, indent + 1, output_lines
#             )
#             pp_lines = []
#             pp = next_pp
#             first = False
#         pp_lines.append(line)
#     # write out the last one
#     write_mainchain_bin_rotamer_entries(pp_lines, first, nchi, indent + 1, output_lines)
#     output_lines.append("".join([" " * indent, "]}\n"]))
#
#
# def write_semi_rotameric_rotamer_entry(
#     line, first, chi_labels, n_rotameric_chi, indent, output_lines
# ):
#     cols = line.split()
#     bb_dihedrals = cols[1:3]
#     output_lines.append(
#         "".join(
#             [
#                 " " * indent,
#                 "" if first else ",",
#                 '{"bb_dihedrals": [',
#                 ", ".join(bb_dihedrals),
#                 '], "chi_probabilities": [\n',
#             ]
#         )
#     )
#
#     prob_start_col = 5 + 3 * n_rotameric_chi
#     prob_cols = cols[prob_start_col:]
#     for i, chi_label in enumerate(chi_labels):
#         output_lines.append(
#             "".join(
#                 [
#                     " " * (indent + 1),
#                     "" if i == 0 else ",",
#                     '{"chi": ',
#                     chi_label,
#                     ', "prob": ',
#                     prob_cols[i],
#                     "}\n",
#                 ]
#             )
#         )
#     output_lines.append("".join([" " * indent, "]}\n"]))
#
#
# def write_semi_rotameric_table_for_rotamer(
#     lines, first, n_rotameric_chi, chi_labels, indent, output_lines
# ):
#     rotamer_slice = slice(4, 4 + n_rotameric_chi)
#     rotamer = lines[0].split()[rotamer_slice]
#     output_lines.append(
#         "".join(
#             [
#                 " " * indent,
#                 "" if first else ",",
#                 '{"rotamer": [',
#                 ", ".join(rotamer),
#                 '], "entries": [\n',
#             ]
#         )
#     )
#     first = True
#     for line in lines:
#         write_semi_rotameric_rotamer_entry(
#             line, first, chi_labels, n_rotameric_chi, indent + 1, output_lines
#         )
#         first = False
#     output_lines.append("".join([" " * indent, "]}\n"]))
#
#
# def write_semi_rotameric_rotamer_definition(line, first, nchi, indent, output_lines):
#     cols = line.split()
#     output_lines.append(
#         "".join(
#             [
#                 " " * indent,
#                 "" if first else ",",
#                 '{"rotamer": [',
#                 ", ".join(cols[0:nchi]),
#                 '], "left": ',
#                 cols[6],
#                 ', "right": ',
#                 cols[8],
#                 "}\n",
#             ]
#         )
#     )
#
#
# def write_aa_semi_rotameric_dunbrack_library(
#     bbdep_rotamer_lines,
#     bbdep_density_lines,
#     bbind_rotamer_def_lines,
#     indent,
#     output_lines,
#     first,
# ):
#     aa3 = None
#     for line in bbdep_rotamer_lines:
#         if len(line) == 0 or line[0] == "#":
#             continue
#         aa3 = line.split()[0]
#         break
#
#     nchi_for_aa = {
#         "CYS": 1,
#         "ASP": 2,
#         "GLU": 3,
#         "PHE": 2,
#         "HIS": 2,
#         "ILE": 2,
#         "LYS": 4,
#         "LEU": 2,
#         "MET": 3,
#         "ASN": 2,
#         "PRO": 3,
#         "GLN": 3,
#         "ARG": 4,
#         "SER": 1,
#         "THR": 1,
#         "VAL": 1,
#         "TRP": 2,
#         "TYR": 2,
#     }
#     nchi = nchi_for_aa[aa3]
#
#     output_lines.append(
#         "".join([" " * indent, "" if first else ",", '{"aa_name": "', aa3, '",\n'])
#     )
#     output_lines.append("".join([" " * (indent + 1), '"mc_dihedral_entries": [\n']))
#
#     lines_by_bb_bin = {}
#     bb_bins = []
#     for line in bbdep_rotamer_lines:
#         if len(line) == 0 or line[0] == "#":
#             continue
#         cols = line.split()
#         phi_psi_bin = (cols[1], cols[2])
#         if phi_psi_bin not in lines_by_bb_bin:
#             lines_by_bb_bin[phi_psi_bin] = []
#             bb_bins.append(phi_psi_bin)
#         lines_by_bb_bin[phi_psi_bin].append(line)
#
#     first = True
#     for phi_psi_bin in bb_bins:
#         write_mainchain_bin_rotamer_entries(
#             lines_by_bb_bin[phi_psi_bin], first, nchi, indent + 2, output_lines
#         )
#         first = False
#     output_lines.append("".join([" " * (indent + 1), "],\n"]))
#
#     n_rotameric_chi_for_aa = {
#         "ASP": 1,
#         "ASN": 1,
#         "GLU": 2,
#         "GLN": 2,
#         "PHE": 1,
#         "TRP": 1,
#         "HIS": 1,
#         "TYR": 1,
#     }
#     n_rotameric_chi = n_rotameric_chi_for_aa[aa3]
#     rot_columns = None
#     lines_for_rotamers = {}
#     last_line = None
#     second_to_last_line = None
#     first_non_comment = True
#     rotamers = []
#     for line in bbdep_density_lines:
#         if len(line) == 0 or line[0] == "#":
#             second_to_last_line = last_line
#             last_line = line
#             continue
#         if first_non_comment:
#             cols = second_to_last_line[1:].split()
#             chi_labels = cols[(5 + 3 * n_rotameric_chi) :]
#             first_non_comment = False
#         if rot_columns is None:
#             rot_columns = slice(4, 4 + n_rotameric_chi)
#         cols = line.split()
#         rot_bin = tuple(cols[rot_columns])
#         if rot_bin not in lines_for_rotamers:
#             lines_for_rotamers[rot_bin] = []
#             rotamers.append(rot_bin)
#         lines_for_rotamers[rot_bin].append(line)
#
#     output_lines.append("".join([" " * (indent + 1), '"semi_rotameric_tables": [\n']))
#     first = True
#     for rotamer in rotamers:
#         write_semi_rotameric_table_for_rotamer(
#             lines_for_rotamers[rotamer],
#             first,
#             n_rotameric_chi,
#             chi_labels,
#             indent + 2,
#             output_lines,
#         )
#         first = False
#     output_lines.append("".join([" " * (indent + 1), "],\n"]))
#
#     output_lines.append("".join([" " * (indent + 1), '"rotamer_definitions": [\n']))
#     first = True
#     for line in bbind_rotamer_def_lines:
#         if len(line) == 0 or line[0] == "#":
#             continue
#         write_semi_rotameric_rotamer_definition(
#             line, first, nchi, indent + 2, output_lines
#         )
#         first = False
#     output_lines.append("".join([" " * (indent + 1), "]\n"]))
#
#     output_lines.append("".join([" " * indent, "}\n"]))
#
#
# def write_json_version_of_dunbrack_rotamer_library(path_to_db_dir):
#     outlines = []
#     outlines.append('{"rotameric_libraries": [\n')
#
#     # Rotameric residues:
#     # CYS, ILE, LYS, LEU, MET, PRO, ARG, SER, THR, VAL
#     lib_files = [
#         (path_to_db_dir + x + ".bbdep.rotamers.lib.gz")
#         for x in ("cys", "ile", "lys", "met", "pro", "arg", "ser", "thr", "val")
#     ]
#     first = True
#     for lib_file in lib_files:
#         print("processing", lib_file)
#         with gzip.GzipFile(lib_file) as fid:
#             lines = [x.decode("utf-8") for x in fid.readlines()]
#         write_aa_rotameric_dunbrack_library(lines, first, 2, outlines)
#         first = False
#
#     outlines.append(" ],\n")
#     outlines.append(' "semi_rotameric_libraries": [\n')
#
#     # Semirotameric residues:
#     # ASP, GLU, PHE, HIS, ASN, GLN, TRP, TYR
#     nchi = {
#         "asp": 2,
#         "glu": 3,
#         "phe": 2,
#         "his": 2,
#         "asn": 2,
#         "gln": 3,
#         "trp": 2,
#         "tyr": 2,
#     }
#     aas = ["asp", "glu", "phe", "his", "asn", "gln", "trp", "tyr"]
#     lib_files = [
#         (
#             path_to_db_dir + x + ".bbdep.rotamers.lib.gz",
#             path_to_db_dir + x + ".bbdep.densities.lib.gz",
#             path_to_db_dir + x + ".bbind.chi" + str(nchi[x]) + ".Definitions.lib.gz",
#         )
#         for x in aas
#     ]
#     first = True
#     for files in lib_files:
#         with gzip.GzipFile(files[0]) as fid:
#             rotamer_lines = [x.decode("utf-8") for x in fid.readlines()]
#         with gzip.GzipFile(files[1]) as fid:
#             density_lines = [x.decode("utf-8") for x in fid.readlines()]
#         with gzip.GzipFile(files[2]) as fid:
#             bbind_rotamer_lines = [x.decode("utf-8") for x in fid.readlines()]
#         print("processing", files[0])
#         write_aa_semi_rotameric_dunbrack_library(
#             rotamer_lines, density_lines, bbind_rotamer_lines, 2, outlines, first
#         )
#         first = False
#
#     outlines.append(" ]\n")
#     outlines.append("}\n")
#
#     return outlines
#
#
# if __name__ == "__main__":
#     ## # let's test some stuff
#     ## rotamer_lines = open("dummy_gln_rotamers.txt").readlines()
#     ## density_lines = open("dummy_gln_densities.txt").readlines()
#     ## rotdef_lines = open("dummy_gln_rotdef.txt").readlines()
#     ## outlines = []
#     ## write_aa_semi_rotameric_dunbrack_library(
#     ##     rotamer_lines,
#     ##     density_lines,
#     ##     rotdef_lines,
#     ##     0,
#     ##     outlines,
#     ##     True)
#     ## with open("dummy_gln_rotlib.json", "w") as fid:
#     ##     fid.writelines(outlines)
#
#     outlines = write_json_version_of_dunbrack_rotamer_library(
#         "/Users/andrew/rosetta/GIT/Rosetta/main/database/rotamer/ExtendedOpt1-5/"
#     )
#     open("dunbrack_library.json", "w").writelines(outlines)


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
