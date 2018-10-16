import numpy
import zarr


def write_rama_table_for_aa(name, prob_table, zgroup):
    table_group = zgroup.create_group(name)
    table_group.array("bb_start", [-180, -180])
    table_group.array("bb_step", [10, 10])
    table_group.array("probabilities", prob_table)


def write_lines_to_zarr(lines, isprepro, zgroup):
    curr_aa = None
    prob_table = numpy.zeros((36, 36), dtype=numpy.float)
    table_names = []
    for line in lines:
        cols = line.split()
        if cols[0] != curr_aa:
            if curr_aa != None:
                write_rama_table_for_aa(table_name, prob_table, zgroup)
                table_names.append(table_name)
                prob_table = numpy.zeros((36, 36), dtype=numpy.float)
            curr_aa = cols[0]
            table_name = "LAA_%s_%s" % (cols[0], ("PREPRO" if isprepro else "STANDARD"))
        phi_ind, psi_ind = (int(x) // 10 + 18 for x in cols[1:3])
        prob_table[psi_ind, phi_ind] = float(cols[3])

    write_rama_table_for_aa(table_name, prob_table, zgroup)
    table_names.append(table_name)
    return table_names


if __name__ == "__main__":

    store = zarr.ZipStore("rama/rama.bin")
    zgroup = zarr.group(store=store)

    names1 = write_lines_to_zarr(open("all.ramaProb").readlines(), False, zgroup)
    names2 = write_lines_to_zarr(open("prepro.ramaProb").readlines(), True, zgroup)

    names1.extend(names2)
    zgroup.attrs.update(tables=names1)

    store.close()
