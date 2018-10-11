import numpy
from decimal import Decimal


def write_table(table, energies, indent, add_trailing_comma, json_lines):
    format = "%10.4f" if energies else "%10.5E"
    nrows = table.shape[0]
    ncols = table.shape[1]
    for i in range(nrows):
        line_entries = []
        if i == 0:
            line_entries.append(" " * indent)
            line_entries.append("[[")
        else:
            line_entries.append(" " * (indent + 1))
            line_entries.append("[")
        for j in range(ncols):
            if j != 0:
                line_entries.append(", ")
            line_entries.append(
                format % (table[i, j] if energies else Decimal(table[i, j]))
            )
            if j == ncols - 1:
                line_entries.append("]")
                if i == nrows - 1:
                    line_entries.append("]")
                    if add_trailing_comma:
                        line_entries.append(",")
                    line_entries.append("\n")
                else:
                    line_entries.append(",\n")
        json_lines.append("".join(line_entries))


def write_lines_as_json(lines, isprepro, trailing_comma):
    curr_aa = None
    json_lines = []
    prob_table = numpy.zeros((36, 36), dtype=numpy.float)
    # energies_table = numpy.zeros((36, 36), dtype=numpy.float)
    for line in lines:
        cols = line.split()
        if cols[0] != curr_aa:
            if curr_aa != None:
                json_lines.append('    "probabilities":\n')
                write_table(prob_table, False, 5, False, json_lines)
                # json_lines.append('   "energies":\n')
                # write_table(energies_table, True, 5, False, json_lines)
                json_lines.append("  },\n")

                prob_table = numpy.zeros((36, 36), dtype=numpy.float)
                # energies_table = numpy.zeros((36, 36), dtype=numpy.float)
            curr_aa = cols[0]
            # json_lines.append( "  - {\n" )
            table_name = "LAA_%s_%s" % (cols[0], ("PREPRO" if isprepro else "STANDARD"))
            json_lines.append('  { "name": "%s",\n' % table_name)
            json_lines.append('    "phi_step": 10,\n')
            json_lines.append('    "psi_step": 10,\n')
            json_lines.append('    "phi_start": -180,\n')
            json_lines.append('    "psi_start": -180,\n')
        phi_ind = int(cols[1]) // 10 + 18
        psi_ind = 35 - (int(cols[2]) // 10 + 18)
        # Column major resembles the image you naturally have
        # of Y decreasing as you move down and X increasing
        # as you move across.
        prob_table[psi_ind, phi_ind] = cols[3]
        # energies_table[psi_ind, phi_ind] = cols[4]

    json_lines.append('    "probabilities":\n')
    write_table(prob_table, False, 5, False, json_lines)
    # json_lines.append('   "energies":\n')
    # write_table(energies_table, True, 5, False, json_lines)
    if trailing_comma:
        json_lines.append("  },\n")
    else:
        json_lines.append("  }\n")

    return json_lines


if __name__ == "__main__":

    json_lines = []
    json_lines.append('{ "tables": [\n')

    json_lines.extend(
        write_lines_as_json(open("all.ramaProb").readlines(), False, True)
    )
    json_lines.extend(
        write_lines_as_json(open("prepro.ramaProb").readlines(), True, False)
    )
    json_lines.append("  ],\n")

    json_lines.append(
        """  "evaluation_mappings": [
    {"condition": "aa.alpha.l.alanine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_ALA_STANDARD"},
    {"condition": "aa.alpha.l.arginine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_ARG_STANDARD"},
    {"condition": "aa.alpha.l.asparagine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_ASN_STANDARD"},
    {"condition": "aa.alpha.l.aspartate,(upper:!aa.alpha.l.proline)", "table_name": "LAA_ASP_STANDARD"},
    {"condition": "aa.alpha.l.cysteine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_CYS_STANDARD"},
    {"condition": "aa.alpha.l.glutamine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_GLN_STANDARD"},
    {"condition": "aa.alpha.l.glutamate,(upper:!aa.alpha.l.proline)", "table_name": "LAA_GLU_STANDARD"},
    {"condition": "aa.alpha.l.glycine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_GLY_STANDARD"},
    {"condition": "aa.alpha.l.histidine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_HIS_STANDARD"},
    {"condition": "aa.alpha.l.isoleucine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_ILE_STANDARD"},
    {"condition": "aa.alpha.l.leucine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_LEU_STANDARD"},
    {"condition": "aa.alpha.l.lysine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_LYS_STANDARD"},
    {"condition": "aa.alpha.l.methionine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_MET_STANDARD"},
    {"condition": "aa.alpha.l.phenylalanine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_PHE_STANDARD"},
    {"condition": "aa.alpha.l.proline,(upper:!aa.alpha.l.proline)", "table_name": "LAA_PRO_STANDARD"},
    {"condition": "aa.alpha.l.serine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_SER_STANDARD"},
    {"condition": "aa.alpha.l.threonine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_THR_STANDARD"},
    {"condition": "aa.alpha.l.tryptophan,(upper:!aa.alpha.l.proline)", "table_name": "LAA_TRP_STANDARD"},
    {"condition": "aa.alpha.l.tyrosine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_TYR_STANDARD"},
    {"condition": "aa.alpha.l.valine,(upper:!aa.alpha.l.proline)", "table_name": "LAA_VAL_STANDARD"},
    {"condition": "aa.alpha.l.alanine,(upper:aa.alpha.l.proline)", "table_name": "LAA_ALA_PREPRO"},
    {"condition": "aa.alpha.l.arginine,(upper:aa.alpha.l.proline)", "table_name": "LAA_ARG_PREPRO"},
    {"condition": "aa.alpha.l.asparagine,(upper:aa.alpha.l.proline)", "table_name": "LAA_ASN_PREPRO"},
    {"condition": "aa.alpha.l.aspartate,(upper:aa.alpha.l.proline)", "table_name": "LAA_ASP_PREPRO"},
    {"condition": "aa.alpha.l.cysteine,(upper:aa.alpha.l.proline)", "table_name": "LAA_CYS_PREPRO"},
    {"condition": "aa.alpha.l.glutamine,(upper:aa.alpha.l.proline)", "table_name": "LAA_GLN_PREPRO"},
    {"condition": "aa.alpha.l.glutamate,(upper:aa.alpha.l.proline)", "table_name": "LAA_GLU_PREPRO"},
    {"condition": "aa.alpha.l.glycine,(upper:aa.alpha.l.proline)", "table_name": "LAA_GLY_PREPRO"},
    {"condition": "aa.alpha.l.histidine,(upper:aa.alpha.l.proline)", "table_name": "LAA_HIS_PREPRO"},
    {"condition": "aa.alpha.l.isoleucine,(upper:aa.alpha.l.proline)", "table_name": "LAA_ILE_PREPRO"},
    {"condition": "aa.alpha.l.leucine,(upper:aa.alpha.l.proline)", "table_name": "LAA_LEU_PREPRO"},
    {"condition": "aa.alpha.l.lysine,(upper:aa.alpha.l.proline)", "table_name": "LAA_LYS_PREPRO"},
    {"condition": "aa.alpha.l.methionine,(upper:aa.alpha.l.proline)", "table_name": "LAA_MET_PREPRO"},
    {"condition": "aa.alpha.l.phenylalanine,(upper:aa.alpha.l.proline)", "table_name": "LAA_PHE_PREPRO"},
    {"condition": "aa.alpha.l.proline,(upper:aa.alpha.l.proline)", "table_name": "LAA_PRO_PREPRO"},
    {"condition": "aa.alpha.l.serine,(upper:aa.alpha.l.proline)", "table_name": "LAA_SER_PREPRO"},
    {"condition": "aa.alpha.l.threonine,(upper:aa.alpha.l.proline)", "table_name": "LAA_THR_PREPRO"},
    {"condition": "aa.alpha.l.tryptophan,(upper:aa.alpha.l.proline)", "table_name": "LAA_TRP_PREPRO"},
    {"condition": "aa.alpha.l.tyrosine,(upper:aa.alpha.l.proline)", "table_name": "LAA_TYR_PREPRO"},
    {"condition": "aa.alpha.l.valine,(upper:aa.alpha.l.proline)", "table_name": "LAA_VAL_PREPRO"}
  ]
"""
    )

    json_lines.append("}\n")

    open("rama.json", "w").writelines(json_lines)
