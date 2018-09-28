import gzip


def write_dunbrack_rotamer_entry(line, first, nchi, indent, output_lines):
    cols = line.split()

    rotamer = tuple(cols[4 : (4 + nchi)])
    prob = cols[8]
    means = tuple(cols[9 : (9 + nchi)])
    stdv = tuple(cols[13 : (13 + nchi)])
    output_lines.append(
        "".join(
            [
                " " * indent,
                "" if first else ",",
                '{"rotamer": [',
                ", ".join(rotamer),
                '], "prob": ',
                prob,
                ', "means": [',
                ", ".join(means),
                '], "stdev": [',
                ", ".join(stdv),
                "]}\n",
            ]
        )
    )


def write_mainchain_bin_rotamer_entries(lines, first, nchi, indent, output_lines):
    phi_psi_bin = tuple(lines[0].split()[1:3])
    output_lines.append(
        "".join(
            [
                " " * indent,
                "" if first else ",",
                '{"bb_dihedrals": [',
                ", ".join(phi_psi_bin),
                '], "sorted_rotamers": [\n',
            ]
        )
    )
    first = True
    for line in lines:
        write_dunbrack_rotamer_entry(line, first, nchi, indent + 1, output_lines)
        first = False
    output_lines.append("".join([" " * indent, "]}\n"]))


def write_aa_rotameric_dunbrack_library(lines, first, indent, output_lines):
    nchi_for_aa = {
        "CYS": 1,
        "ASP": 1,
        "GLU": 2,
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
    for line in lines:
        if len(line) == 0 or line[0] == "#":
            continue
        aa3 = line.split()[0]
        pp = tuple(line.split()[1:3])
        break
    nchi = nchi_for_aa[aa3]
    output_lines.append(
        "".join(
            [
                " " * indent,
                "" if first else ",",
                '{"aa_name": "',
                aa3,
                '", "mc_dihedral_entries": [\n',
            ]
        )
    )
    pp_lines = []
    first = True
    for line in lines:
        if len(line) == 0 or line[0] == "#":
            continue
        next_pp = tuple(line.split()[1:3])
        if pp != next_pp:
            write_mainchain_bin_rotamer_entries(
                pp_lines, first, nchi, indent + 1, output_lines
            )
            pp_lines = []
            pp = next_pp
            first = False
        pp_lines.append(line)
    # write out the last one
    write_mainchain_bin_rotamer_entries(pp_lines, first, nchi, indent + 1, output_lines)
    output_lines.append("".join([" " * indent, "]}\n"]))


def write_semi_rotameric_rotamer_entry(
    line, first, chi_labels, n_rotameric_chi, indent, output_lines
):
    cols = line.split()
    bb_dihedrals = cols[1:3]
    output_lines.append(
        "".join(
            [
                " " * indent,
                "" if first else ",",
                '{"bb_dihedrals": [',
                ", ".join(bb_dihedrals),
                '], "chi_probabilities": [\n',
            ]
        )
    )

    prob_start_col = 5 + 3 * n_rotameric_chi
    prob_cols = cols[prob_start_col:]
    for i, chi_label in enumerate(chi_labels):
        output_lines.append(
            "".join(
                [
                    " " * (indent + 1),
                    "" if i == 0 else ",",
                    '{"chi": ',
                    chi_label,
                    ', "prob": ',
                    prob_cols[i],
                    "}\n",
                ]
            )
        )
    output_lines.append("".join([" " * indent, "]}\n"]))


def write_semi_rotameric_table_for_rotamer(
    lines, first, n_rotameric_chi, chi_labels, indent, output_lines
):
    rotamer_slice = slice(4, 4 + n_rotameric_chi)
    rotamer = lines[0].split()[rotamer_slice]
    output_lines.append(
        "".join(
            [
                " " * indent,
                "" if first else ",",
                '{"rotamer": [',
                ", ".join(rotamer),
                '], "entries": [\n',
            ]
        )
    )
    first = True
    for line in lines:
        write_semi_rotameric_rotamer_entry(
            line, first, chi_labels, n_rotameric_chi, indent + 1, output_lines
        )
        first = False
    output_lines.append("".join([" " * indent, "]}\n"]))


def write_semi_rotameric_rotamer_definition(line, first, nchi, indent, output_lines):
    cols = line.split()
    output_lines.append(
        "".join(
            [
                " " * indent,
                "" if first else ",",
                '{"rotamer": [',
                ", ".join(cols[0:nchi]),
                '], "left": ',
                cols[6],
                ', "right": ',
                cols[8],
                "}\n",
            ]
        )
    )


def write_aa_semi_rotameric_dunbrack_library(
    bbdep_rotamer_lines,
    bbdep_density_lines,
    bbind_rotamer_def_lines,
    indent,
    output_lines,
    first,
):
    aa3 = None
    for line in bbdep_rotamer_lines:
        if len(line) == 0 or line[0] == "#":
            continue
        aa3 = line.split()[0]
        break

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
    nchi = nchi_for_aa[aa3]

    output_lines.append(
        "".join([" " * indent, "" if first else ",", '{"aa_name": "', aa3, '",\n'])
    )
    output_lines.append("".join([" " * (indent + 1), '"mc_dihedral_entries": [\n']))

    lines_by_bb_bin = {}
    bb_bins = []
    for line in bbdep_rotamer_lines:
        if len(line) == 0 or line[0] == "#":
            continue
        cols = line.split()
        phi_psi_bin = (cols[1], cols[2])
        if phi_psi_bin not in lines_by_bb_bin:
            lines_by_bb_bin[phi_psi_bin] = []
            bb_bins.append(phi_psi_bin)
        lines_by_bb_bin[phi_psi_bin].append(line)

    first = True
    for phi_psi_bin in bb_bins:
        write_mainchain_bin_rotamer_entries(
            lines_by_bb_bin[phi_psi_bin], first, nchi, indent + 2, output_lines
        )
        first = False
    output_lines.append("".join([" " * (indent + 1), "],\n"]))

    n_rotameric_chi_for_aa = {
        "ASP": 1,
        "ASN": 1,
        "GLU": 2,
        "GLN": 2,
        "PHE": 1,
        "TRP": 1,
        "HIS": 1,
        "TYR": 1,
    }
    n_rotameric_chi = n_rotameric_chi_for_aa[aa3]
    rot_columns = None
    lines_for_rotamers = {}
    last_line = None
    second_to_last_line = None
    first_non_comment = True
    rotamers = []
    for line in bbdep_density_lines:
        if len(line) == 0 or line[0] == "#":
            second_to_last_line = last_line
            last_line = line
            continue
        if first_non_comment:
            cols = second_to_last_line[1:].split()
            chi_labels = cols[(5 + 3 * n_rotameric_chi) :]
            first_non_comment = False
        if rot_columns is None:
            rot_columns = slice(4, 4 + n_rotameric_chi)
        cols = line.split()
        rot_bin = tuple(cols[rot_columns])
        if rot_bin not in lines_for_rotamers:
            lines_for_rotamers[rot_bin] = []
            rotamers.append(rot_bin)
        lines_for_rotamers[rot_bin].append(line)

    output_lines.append("".join([" " * (indent + 1), '"semi_rotameric_tables": [\n']))
    first = True
    for rotamer in rotamers:
        write_semi_rotameric_table_for_rotamer(
            lines_for_rotamers[rotamer],
            first,
            n_rotameric_chi,
            chi_labels,
            indent + 2,
            output_lines,
        )
        first = False
    output_lines.append("".join([" " * (indent + 1), "],\n"]))

    output_lines.append("".join([" " * (indent + 1), '"rotamer_definitions": [\n']))
    first = True
    for line in bbind_rotamer_def_lines:
        if len(line) == 0 or line[0] == "#":
            continue
        write_semi_rotameric_rotamer_definition(
            line, first, nchi, indent + 2, output_lines
        )
        first = False
    output_lines.append("".join([" " * (indent + 1), "]\n"]))

    output_lines.append("".join([" " * indent, "}\n"]))


def write_json_version_of_dunbrack_rotamer_library(path_to_db_dir):
    outlines = []
    outlines.append('{"rotameric_libraries": [\n')

    # Rotameric residues:
    # CYS, ILE, LYS, LEU, MET, PRO, ARG, SER, THR, VAL
    lib_files = [
        (path_to_db_dir + x + ".bbdep.rotamers.lib.gz")
        for x in ("cys", "ile", "lys", "met", "pro", "arg", "ser", "thr", "val")
    ]
    first = True
    for lib_file in lib_files:
        print("processing", lib_file)
        with gzip.GzipFile(lib_file) as fid:
            lines = [x.decode("utf-8") for x in fid.readlines()]
        write_aa_rotameric_dunbrack_library(lines, first, 2, outlines)
        first = False

    outlines.append(" ],\n")
    outlines.append(' "semi_rotameric_libraries": [\n')

    # Semirotameric residues:
    # ASP, GLU, PHE, HIS, ASN, GLN, TRP, TYR
    nchi = {
        "asp": 2,
        "glu": 3,
        "phe": 2,
        "his": 2,
        "asn": 2,
        "gln": 3,
        "trp": 2,
        "tyr": 2,
    }
    aas = ["asp", "glu", "phe", "his", "asn", "gln", "trp", "tyr"]
    lib_files = [
        (
            path_to_db_dir + x + ".bbdep.rotamers.lib.gz",
            path_to_db_dir + x + ".bbdep.densities.lib.gz",
            path_to_db_dir + x + ".bbind.chi" + str(nchi[x]) + ".Definitions.lib.gz",
        )
        for x in aas
    ]
    first = True
    for files in lib_files:
        with gzip.GzipFile(files[0]) as fid:
            rotamer_lines = [x.decode("utf-8") for x in fid.readlines()]
        with gzip.GzipFile(files[1]) as fid:
            density_lines = [x.decode("utf-8") for x in fid.readlines()]
        with gzip.GzipFile(files[2]) as fid:
            bbind_rotamer_lines = [x.decode("utf-8") for x in fid.readlines()]
        print("processing", files[0])
        write_aa_semi_rotameric_dunbrack_library(
            rotamer_lines, density_lines, bbind_rotamer_lines, 2, outlines, first
        )
        first = False

    outlines.append(" ]\n")
    outlines.append("}\n")

    return outlines


if __name__ == "__main__":
    ## # let's test some stuff
    ## rotamer_lines = open("dummy_gln_rotamers.txt").readlines()
    ## density_lines = open("dummy_gln_densities.txt").readlines()
    ## rotdef_lines = open("dummy_gln_rotdef.txt").readlines()
    ## outlines = []
    ## write_aa_semi_rotameric_dunbrack_library(
    ##     rotamer_lines,
    ##     density_lines,
    ##     rotdef_lines,
    ##     0,
    ##     outlines,
    ##     True)
    ## with open("dummy_gln_rotlib.json", "w") as fid:
    ##     fid.writelines(outlines)

    outlines = write_json_version_of_dunbrack_rotamer_library(
        "/Users/andrew/rosetta/GIT/Rosetta/main/database/rotamer/ExtendedOpt1-5/"
    )
    open("dunbrack_library.json", "w").writelines(outlines)
