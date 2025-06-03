import numpy
import torch
import attr
import os
import pickle

from tmol.pack.datatypes import PackerEnergyTables
from tmol.pack.simulated_annealing import run_simulated_annealing
from tmol.utility.cumsum import exclusive_cumsum


def load_ig_from_file(fname):
    with open("1ubq_ig", "rb") as f:
        return pickle.load(f)
    store = zarr.ZipStore(fname)
    zgroup = zarr.group(store=store)
    nres = zgroup.attrs["nres"]
    oneb_energies = {}
    # restype_groups = {}
    twob_energies = {}
    for i in range(1, nres + 1):
        oneb_arrname = "%d" % i
        restype_group_arrname = "%d_rtgroups" % i
        oneb_energies[oneb_arrname] = numpy.array(zgroup[oneb_arrname], dtype=float)
        # restype_groups[oneb_arrname] = numpy.array(
        #     zgroup[restype_group_arrname], dtype=int
        # )
        for j in range(i + 1, nres + 1):
            twob_arrname = "%d-%d" % (i, j)
            if twob_arrname in zgroup:
                twob_energies[twob_arrname] = numpy.array(
                    zgroup[twob_arrname], dtype=float
                )
    return oneb_energies, twob_energies


def test_load_ig():
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    assert len(oneb) == 76
    nrots = numpy.zeros((76,), dtype=int)
    for i in range(76):
        arrname = "%d" % (i + 1)
        nrots[i] = oneb[arrname].shape[0]
    for i in range(76):
        for j in range(i + 1, 76):
            arrname = "%d-%d" % (i + 1, j + 1)
            if arrname in twob:
                assert nrots[i] == twob[arrname].shape[0]
                assert nrots[j] == twob[arrname].shape[1]


def aa_neighb_nonzero_submatrix(twob, rtg1, rtg2):
    # rtg1 = exclusive_cumsum(rtg1_start)
    # rtg2 = exclusive_cumsum(rtg2_start)

    rtg1_start = numpy.concatenate((numpy.ones(1, dtype=int), rtg1[1:] - rtg1[:-1]))
    rtg2_start = numpy.concatenate((numpy.ones(1, dtype=int), rtg2[1:] - rtg2[:-1]))

    n_rtg1 = numpy.sum(rtg1_start)
    n_rtg2 = numpy.sum(rtg2_start)

    rtg1_offsets = numpy.nonzero(rtg1_start)[0]
    rtg2_offsets = numpy.nonzero(rtg2_start)[0]

    rtg_nrots1 = numpy.concatenate(
        (
            rtg1_offsets[1:] - rtg1_offsets[:-1],
            numpy.full((1,), rtg1_start.shape[0] - rtg1_offsets[-1], dtype=int),
        )
    )
    rtg_nrots2 = numpy.concatenate(
        (
            rtg2_offsets[1:] - rtg2_offsets[:-1],
            numpy.full((1,), rtg2_start.shape[0] - rtg2_offsets[-1], dtype=int),
        )
    )
    # print(rtg_nrots1)
    # print(rtg_nrots2)

    fine_offsets = numpy.full((n_rtg1, n_rtg2), -1, dtype=int)
    count = 0
    for i in range(n_rtg1):
        i_slice = slice(rtg1_offsets[i], (rtg1_offsets[i] + rtg_nrots1[i]))
        for j in range(n_rtg2):
            j_slice = slice(rtg2_offsets[j], (rtg2_offsets[j] + rtg_nrots2[j]))
            e2b_slice = twob[i_slice, j_slice]
            # print(i, rtg_nrots1[i], j, rtg_nrots2[j], e2b_slice.shape)
            assert (rtg_nrots1[i], rtg_nrots2[j]) == e2b_slice.shape
            if numpy.any(e2b_slice != 0):
                fine_offsets[i, j] = count
                count += rtg_nrots1[i] * rtg_nrots2[j]
    rtg_sparse_matrix = numpy.zeros((count,), dtype=float)
    for i in range(n_rtg1):
        i_slice = slice(rtg1_offsets[i], (rtg1_offsets[i] + rtg_nrots1[i]))
        for j in range(n_rtg2):
            j_slice = slice(rtg2_offsets[j], (rtg2_offsets[j] + rtg_nrots2[j]))
            ij_offset = fine_offsets[i, j]
            if ij_offset >= 0:
                e2b_slice = twob[i_slice, j_slice].reshape(-1)
                insert_slice = slice(
                    ij_offset, (ij_offset + rtg_nrots1[i] * rtg_nrots2[j])
                )
                rtg_sparse_matrix[insert_slice] = e2b_slice
    return fine_offsets, rtg_sparse_matrix


def count_aa_sparse_memory_usage(oneb, restype_groups, twob):
    nres = len(oneb)

    count_sparse = 0
    count_dense = 0
    count_nonzero = 0
    for i in range(nres):
        for j in range(i + 1, nres):
            one_name = "{}".format(i + 1)
            two_name = "{}".format(j + 1)
            onetwo_name = "{}-{}".format(i + 1, j + 1)
            if onetwo_name in twob:
                onetwo_twob = twob[onetwo_name]
                fine_offsets, rtg_sparse_matrix = aa_neighb_nonzero_submatrix(
                    onetwo_twob, restype_groups[one_name], restype_groups[two_name]
                )
                count_dense += onetwo_twob.shape[0] * onetwo_twob.shape[1]
                count_sparse += rtg_sparse_matrix.shape[0]
                count_nonzero += numpy.nonzero(rtg_sparse_matrix)[0].shape[0]
    return count_dense, count_sparse, count_nonzero


def dont_test_nonzero_submatrix():
    fname = "1ubq_redes_noex.zarr"
    oneb, restype_groups, twob = load_ig_from_file(fname)

    dense, sparse, nonzero = count_aa_sparse_memory_usage(oneb, restype_groups, twob)
    print(dense, sparse, nonzero)


def dont_test_aasparse_mat_repack():
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs2/repack/" + fname + "_repack_noex.zarr"
        assert os.path.isfile(path_to_zarr_file)
        oneb, restype_groups, twob = load_ig_from_file(path_to_zarr_file)
        dense, sparse, nonzero = count_aa_sparse_memory_usage(
            oneb, restype_groups, twob
        )
        print(dense, sparse, nonzero, nonzero / dense, sparse / dense, nonzero / sparse)


def dont_test_aasparse_mat_redes_ex1ex2():
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs2/redes_ex1ex2/" + fname + "_redes_ex1ex2.zarr"
        assert os.path.isfile(path_to_zarr_file)
        oneb, restype_groups, twob = load_ig_from_file(path_to_zarr_file)
        dense, sparse, nonzero = count_aa_sparse_memory_usage(
            oneb, restype_groups, twob
        )
        print(dense, sparse, nonzero, nonzero / dense, sparse / dense, nonzero / sparse)


def find_nonzero_submatrix_chunks(twob, chunk_size):

    n_rots1 = twob.shape[0]
    n_rots2 = twob.shape[1]
    n_chunks1 = int((n_rots1 - 1) // chunk_size + 1)
    n_chunks2 = int((n_rots2 - 1) // chunk_size + 1)

    # fine_offsets = numpy.full((n_chunks1, n_chunks2), -1, dtype=int)
    chunk_pair_nenergies = numpy.full((n_chunks1, n_chunks2), 0, dtype=int)
    for i in range(n_chunks1):
        i_nrots = min(n_rots1 - i * chunk_size, chunk_size)
        i_slice = slice(i * chunk_size, i * chunk_size + i_nrots)
        for j in range(n_chunks2):
            j_nrots = min(n_rots2 - j * chunk_size, chunk_size)
            j_slice = slice(j * chunk_size, j * chunk_size + j_nrots)
            e2b_slice = twob[i_slice, j_slice]
            # print(i, rtg_nrots1[i], j, rtg_nrots2[j], e2b_slice.shape)
            assert (i_nrots, j_nrots) == e2b_slice.shape
            if numpy.any(e2b_slice != 0):
                chunk_pair_nenergies[i, j] = i_nrots * j_nrots
    return chunk_pair_nenergies


def chunk_nonzero_submatrix(twob, chunk_pair_nenergies, chunk_size):
    n_rots1 = twob.shape[0]
    n_rots2 = twob.shape[1]
    n_chunks1 = int((n_rots1 - 1) // chunk_size + 1)
    n_chunks2 = int((n_rots2 - 1) // chunk_size + 1)

    nenergies = numpy.sum(chunk_pair_nenergies)
    fine_offsets = exclusive_cumsum(chunk_pair_nenergies.reshape(-1)).reshape(
        chunk_pair_nenergies.shape
    )
    fine_offsets[chunk_pair_nenergies == 0] = -1
    rtg_sparse_matrix = numpy.zeros((nenergies,), dtype=float)
    for i in range(n_chunks1):
        i_nrots = min(n_rots1 - i * chunk_size, chunk_size)
        i_slice = slice(i * chunk_size, i * chunk_size + i_nrots)
        for j in range(n_chunks2):
            j_nrots = min(n_rots2 - j * chunk_size, chunk_size)
            j_slice = slice(j * chunk_size, j * chunk_size + j_nrots)
            ij_offset = fine_offsets[i, j]
            if ij_offset >= 0:
                e2b_slice = twob[i_slice, j_slice].reshape(-1)
                insert_slice = slice(ij_offset, (ij_offset + i_nrots * j_nrots))
                rtg_sparse_matrix[insert_slice] = e2b_slice
    return fine_offsets, rtg_sparse_matrix


def count_chunk_sparse_memory_usage(oneb, twob, chunk_size):
    nres = len(oneb)

    count_sparse = 0
    count_dense = 0
    count_nonzero = 0
    for i in range(nres):
        for j in range(i + 1, nres):
            one_name = "{}".format(i + 1)
            two_name = "{}".format(j + 1)
            onetwo_name = "{}-{}".format(i + 1, j + 1)
            if onetwo_name in twob:
                onetwo_twob = twob[onetwo_name]
                chunk_pair_nenergies = find_nonzero_submatrix_chunks(twob, chunk_size)

                # fine_offsets, rtg_sparse_matrix = chunk_nonzero_submatrix(
                #     onetwo_twob, chunk_pair_nenergies, chunk_size
                # )
                count_dense += onetwo_twob.shape[0] * onetwo_twob.shape[1]
                count_sparse += numpy.sum(
                    chunk_pair_nenergies
                )  # rtg_sparse_matrix.shape[0]
                count_nonzero += numpy.nonzero(rtg_sparse_matrix)[0].shape[0]
    return count_dense, count_sparse, count_nonzero


def dont_test_aasparse_mat_repack():
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs2/repack/" + fname + "_repack_noex.zarr"
        assert os.path.isfile(path_to_zarr_file)
        oneb, restype_groups, twob = load_ig_from_file(path_to_zarr_file)
        dense, sparse, nonzero = count_aa_sparse_memory_usage(
            oneb, restype_groups, twob
        )
        print(dense, sparse, nonzero, nonzero / dense, sparse / dense, nonzero / sparse)


def test_aasparse_mat_redes_ex1ex2():
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    results = {}
    for fname in fnames:
        print(fname)
        path_to_zarr_file = "zarr_igs2/redes_ex1ex2/" + fname + "_redes_ex1ex2.zarr"
        assert os.path.isfile(path_to_zarr_file)
        oneb, twob = load_ig_from_file(path_to_zarr_file)
        for chunk in [8, 16, 32, 64]:
            results[(fname, chunk)] = count_chunk_sparse_memory_usage(oneb, twob, chunk)
            print(results[(fname, chunk)])
    print()
    print()
    for chunk in [8, 16, 32, 64]:
        print(chunk)
        for fname in fnames:
            dense, sparse, nonzero = results[(fname, chunk)]
            print(
                dense,
                sparse,
                nonzero,
                nonzero / dense,
                sparse / dense,
                nonzero / sparse,
            )


def count_table_size(twob, restype_groups):
    rtg_start = [1] + restype
    count = 0
    for tabname in twob:
        shape = twob[tabname].shape
        count += shape[0] * shape[1]
    return count


def create_twobody_energy_table(oneb, twob):
    nres = len(oneb)
    offsets = numpy.zeros((nres, nres), dtype=numpy.int64)
    nenergies = numpy.zeros((nres, nres), dtype=int)
    nrotamers_for_res = numpy.array(
        [oneb["{}".format(i + 1)].shape[0] for i in range(nres)], dtype=int
    )
    nrots_total = numpy.sum(nrotamers_for_res)
    oneb_offsets = exclusive_cumsum(nrotamers_for_res)

    energy1b = numpy.zeros(nrots_total, dtype=float)
    res_for_rot = numpy.zeros(nrots_total, dtype=int)
    for i in range(nres):
        tablename = "{}".format(i + 1)
        table = oneb[tablename]
        start = oneb_offsets[i]
        energy1b[(start) : (start + table.shape[0])] = table
        res_for_rot[(start) : (start + table.shape[0])] = i

    for i in range(nres):
        for j in range(i + 1, nres):
            tabname = "{}-{}".format(i + 1, j + 1)
            if tabname in twob:
                nenergies[i, j] = nrotamers_for_res[i] * nrotamers_for_res[j]
                nenergies[j, i] = nrotamers_for_res[i] * nrotamers_for_res[j]

    twob_offsets = exclusive_cumsum(nenergies.reshape(-1)).reshape(nenergies.shape)
    n_rpes_total = numpy.sum(nenergies)
    energy2b = numpy.zeros(n_rpes_total, dtype=float)
    for i in range(nres):
        for j in range(i + 1, nres):
            if nenergies[i, j] == 0:
                continue
            tabname = "{}-{}".format(i + 1, j + 1)
            table = twob[tabname]
            start_ij = twob_offsets[i, j]
            extent = nenergies[i, j]
            energy2b[start_ij : (start_ij + extent)] = table.reshape(-1)
            start_ji = twob_offsets[j, i]
            energy2b[start_ji : (start_ji + extent)] = table.T.reshape(-1)

    return PackerEnergyTables(
        nrotamers_for_res=nrotamers_for_res,
        oneb_offsets=oneb_offsets,
        res_for_rot=res_for_rot,
        nenergies=nenergies,
        twob_offsets=twob_offsets,
        energy1b=energy1b,
        energy2b=energy2b,
    )


def create_chunk_twobody_energy_table(oneb, twob, chunk_size):
    nres = len(oneb)
    offsets = numpy.zeros((nres, nres), dtype=numpy.int64)
    nrotamers_for_res = numpy.array(
        [oneb["{}".format(i + 1)].shape[0] for i in range(nres)], dtype=int
    )
    nrots_total = numpy.sum(nrotamers_for_res)
    oneb_offsets = exclusive_cumsum(nrotamers_for_res)

    energy1b = numpy.zeros(nrots_total, dtype=float)
    res_for_rot = numpy.zeros(nrots_total, dtype=int)
    for i in range(nres):
        tablename = "{}".format(i + 1)
        table = oneb[tablename]
        start = oneb_offsets[i]
        energy1b[(start) : (start + table.shape[0])] = table
        res_for_rot[(start) : (start + table.shape[0])] = i

    # sparse_tables = {}
    # fine_offsets = {}
    chunk_pair_nenergies = {}
    respair_nenergies = numpy.zeros((nres, nres), dtype=int)
    respair_nchunkpairs = numpy.zeros((nres, nres), dtype=int)
    for i in range(nres):
        for j in range(i + 1, nres):
            tabname = "{}-{}".format(i + 1, j + 1)
            if tabname in twob:
                ij_chunk_pair_nenergies = find_nonzero_submatrix_chunks(
                    twob[tabname], chunk_size
                )
                chunk_pair_nenergies[(i, j)] = ij_chunk_pair_nenergies
                ij_nenergies = numpy.sum(ij_chunk_pair_nenergies)
                respair_nenergies[i, j] = ij_nenergies
                respair_nenergies[j, i] = ij_nenergies
                ij_n_sparse_pairs = (
                    ij_chunk_pair_nenergies.shape[0] * ij_chunk_pair_nenergies.shape[1]
                )
                respair_nchunkpairs[i, j] = ij_n_sparse_pairs
                respair_nchunkpairs[j, i] = ij_n_sparse_pairs

    twob_offsets = exclusive_cumsum(respair_nenergies.reshape(-1)).reshape(
        respair_nenergies.shape
    )
    chunk_offset_offsets = exclusive_cumsum(respair_nchunkpairs.reshape(-1)).reshape(
        respair_nchunkpairs.shape
    )

    n_rpes_total = numpy.sum(respair_nenergies)
    n_chunk_offsets_total = numpy.sum(respair_nchunkpairs)

    energy2b = numpy.zeros(n_rpes_total, dtype=float)
    fine_chunk_offsets = numpy.zeros(n_chunk_offsets_total, dtype=int)

    for i in range(nres):
        i_nrotamers = nrotamers_for_res[i]
        i_nchunks = int((i_nrotamers - 1) // chunk_size + 1)
        for j in range(i + 1, nres):
            if respair_nenergies[i, j] == 0:
                continue
            tabname = "{}-{}".format(i + 1, j + 1)
            ij_twob = twob[tabname]

            j_nrotamers = nrotamers_for_res[j]
            j_nchunks = int((j_nrotamers - 1) // chunk_size + 1)

            start_ij = twob_offsets[i, j]
            start_ji = twob_offsets[j, i]

            ij_chunk_pair_nenergies = chunk_pair_nenergies[(i, j)]

            assert (i_nchunks, j_nchunks) == ij_chunk_pair_nenergies.shape

            ij_fine_offsets, ij_sparse_matrix = chunk_nonzero_submatrix(
                ij_twob, ij_chunk_pair_nenergies, chunk_size
            )

            ji_fine_offsets, ji_sparse_matrix = chunk_nonzero_submatrix(
                ij_twob.T, ij_chunk_pair_nenergies.T, chunk_size
            )

            ij_n_chunk_pairs = i_nchunks * j_nchunks

            ij_chunk_offset_offset = chunk_offset_offsets[i, j]
            ij_chunk_offset_slice = slice(
                ij_chunk_offset_offset, ij_chunk_offset_offset + ij_n_chunk_pairs
            )
            fine_chunk_offsets[ij_chunk_offset_slice] = ij_fine_offsets.reshape(-1)

            ji_chunk_offset_offset = chunk_offset_offsets[j, i]
            ji_chunk_offset_slice = slice(
                ji_chunk_offset_offset, ji_chunk_offset_offset + ij_n_chunk_pairs
            )
            fine_chunk_offsets[ji_chunk_offset_slice] = ji_fine_offsets.reshape(-1)

            ij_e2b_offset = twob_offsets[i, j]
            ij_e2b_slice = slice(
                ij_e2b_offset, ij_e2b_offset + ij_sparse_matrix.shape[0]
            )
            energy2b[ij_e2b_slice] = ij_sparse_matrix

            ji_e2b_offset = twob_offsets[j, i]
            ji_e2b_slice = slice(
                ji_e2b_offset, ji_e2b_offset + ji_sparse_matrix.shape[0]
            )
            energy2b[ji_e2b_slice] = ji_sparse_matrix

    chunk_size = numpy.full((1,), chunk_size, dtype=int)

    return PackerEnergyTables(
        nrotamers_for_res=nrotamers_for_res,
        oneb_offsets=oneb_offsets,
        res_for_rot=res_for_rot,
        respair_nenergies=respair_nenergies,
        chunk_size=chunk_size,
        chunk_offset_offsets=chunk_offset_offsets,
        twob_offsets=twob_offsets,
        fine_chunk_offsets=fine_chunk_offsets,
        energy1b=energy1b,
        energy2b=energy2b,
    )


def test_energy_table_construction():
    fname = "1ubq_redes_noex.zarr"
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    chunk_size = 16
    energy_tables = create_chunk_twobody_energy_table(oneb, twob, chunk_size)
    et = energy_tables

    nrots_total = et.res_for_rot.shape[0]
    # pick two residues, 12 and 14
    assert "12-14" in twob

    i_res_nrots = et.nrotamers_for_res[11]
    j_res_nrots = et.nrotamers_for_res[13]

    i_nchunks = (i_res_nrots - 1) // chunk_size + 1
    j_nchunks = (j_res_nrots - 1) // chunk_size + 1

    for i in range(et.oneb_offsets[11], et.oneb_offsets[11] + et.nrotamers_for_res[11]):
        ires = et.res_for_rot[i]
        assert ires == 11
        i_rot_on_res = i - et.oneb_offsets[ires]
        for j in range(
            et.oneb_offsets[13], et.oneb_offsets[13] + et.nrotamers_for_res[13]
        ):
            jres = et.res_for_rot[j]
            assert jres == 13
            j_rot_on_res = j - et.oneb_offsets[jres]
            if et.respair_nenergies[ires, jres] == 0:
                continue

            i_chunk = i_rot_on_res // chunk_size
            j_chunk = j_rot_on_res // chunk_size
            i_rot_in_chunk = i_rot_on_res - chunk_size * i_chunk
            j_rot_in_chunk = j_rot_on_res - chunk_size * j_chunk
            i_chunk_size = min(
                chunk_size, et.nrotamers_for_res[11] - chunk_size * i_chunk
            )
            j_chunk_size = min(
                chunk_size, et.nrotamers_for_res[13] - chunk_size * j_chunk
            )

            ij_chunk_offset = et.chunk_offset_offsets[ires, jres]
            ji_chunk_offset = et.chunk_offset_offsets[jres, ires]

            ij_chunk_start = et.fine_chunk_offsets[
                ij_chunk_offset + i_chunk * j_nchunks + j_chunk
            ]
            ji_chunk_start = et.fine_chunk_offsets[
                ji_chunk_offset + j_chunk * i_nchunks + i_chunk
            ]

            ij_energy = et.energy2b[
                et.twob_offsets[ires, jres]
                + ij_chunk_start
                + i_rot_in_chunk * j_chunk_size
                + j_rot_in_chunk
            ]
            ji_energy = et.energy2b[
                et.twob_offsets[jres, ires]
                + ji_chunk_start
                + j_rot_in_chunk * i_chunk_size
                + i_rot_in_chunk
            ]

            assert ij_energy == ji_energy  # exact equality ok since they are copies


def test_run_sim_annealing():
    torch_device = torch.device("cuda")

    fname = "1ubq_redes_noex.zarr"
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    chunk_size = 16
    et = create_chunk_twobody_energy_table(oneb, twob, chunk_size)

    print("nrotamers", et.res_for_rot.shape[0])
    et_dev = et.to(torch_device)

    scores, rotamer_assignments = run_simulated_annealing(et_dev)

    sort_scores, sort_inds = scores[0, :].sort()
    nkeep = min(scores.shape[0], 20)
    best_scores = sort_scores[0:nkeep].cpu()
    best_score_inds = sort_inds[0:nkeep]
    best_rot_assignments = rotamer_assignments[best_score_inds, :].cpu()

    scores = best_scores.cpu()
    rotamer_assignments = best_rot_assignments.cpu()

    # scores = scores[0:nkeep].cpu()
    # rotamer_assignments = rotamer_assignments[0:nkeep, :].cpu()

    # print("scores", scores, best_scores)
    # print("rotamer_assignments", rotamer_assignments.shape)
    # print("assignment 0", rotamer_assignments[0,0:20])
    # print("sorted assignment 0", best_rot_assignments[0,0:20])

    validated_scores = torch.ops.tmol.validate_energies(
        et.nrotamers_for_res,
        et.oneb_offsets,
        et.res_for_rot,
        et.respair_nenergies,
        et.chunk_size,
        et.chunk_offset_offsets,
        et.twob_offsets,
        et.fine_chunk_offsets,
        et.energy1b,
        et.energy2b,
        rotamer_assignments,
    )

    print("validated scores?", validated_scores)
    torch.testing.assert_allclose(scores, validated_scores)


def test_run_sim_annealing_on_repacking_jobs():
    chunk_size = 16
    torch_device = torch.device("cuda")
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    # fnames = ["1u36FHA"]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs/repack/" + fname + "_repack.zarr"
        oneb, twob = load_ig_from_file(path_to_zarr_file)
        et = create_chunk_twobody_energy_table(oneb, twob, chunk_size)
        # print("nrotamers", et.res_for_rot.shape[0])
        # print("table size", count_table_size(twob))
        et_dev = et.to(torch_device)

        # print("running sim annealing on", fname)
        scores, rotamer_assignments = run_simulated_annealing(et_dev)
        print("scores", scores)

        scores_temp = scores
        # scores = scores.cpu().numpy()
        print("scores again", scores)
        numpy.set_printoptions(threshold=1e5)
        # print("scores", fname)
        # for i in range(scores.shape[1]):
        #    print(scores[0, i], scores[1, i])

        # scores = scores_temp[1, :]
        scores = scores[0, :]
        sort_scores, sort_inds = scores.sort()
        nkeep = min(scores.shape[0], 5)
        best_scores = sort_scores[0:nkeep]
        best_score_inds = sort_inds[0:nkeep]
        best_rot_assignments = rotamer_assignments[best_score_inds, :]

        scores = best_scores.cpu()

        rotamer_assignments = best_rot_assignments.cpu()
        # print("scores", " ".join([str(scores[i].item()) for i in range(scores.shape[0])]))

        validated_scores = torch.ops.tmol.validate_energies(
            et.nrotamers_for_res,
            et.oneb_offsets,
            et.res_for_rot,
            et.respair_nenergies,
            et.chunk_size,
            et.chunk_offset_offsets,
            et.twob_offsets,
            et.fine_chunk_offsets,
            et.energy1b,
            et.energy2b,
            rotamer_assignments,
        )

        # print("validated scores?", validated_scores)
        torch.testing.assert_allclose(scores, validated_scores)


def test_run_sim_annealing_on_redes_ex1ex2_jobs():
    chunk_size = 16
    torch_device = torch.device("cuda")
    fnames = [
        "1wzbFHA",
        "1qtxFHB",
        "1kd8FHB",
        "1ojhFHA",
        "1ff4FHA",
        "1vmgFHA",
        "1u36FHA",
        "1w0nFHA",
    ]
    # fnames = ["1w0nFHA"]
    # fnames = ["1u36FHA", "1w0nFHA"]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs/redes_ex1ex2/" + fname + "_redes_ex1ex2.zarr"
        oneb, twob = load_ig_from_file(path_to_zarr_file)
        # print("table size", count_table_size(twob))
        et = create_chunk_twobody_energy_table(oneb, twob, chunk_size)
        # print("energy2b", et.energy2b.shape[0])
        # print("nrotamers", et.res_for_rot.shape[0])
        # nz = torch.nonzero(et.energy2b)
        # big = torch.nonzero(et.energy2b > 5)
        # print(fname, "number non-zero enties in energy2b:", nz.shape[0] / 2, "big", big.shape[0] / 2, "vs",
        #       et.energy2b.shape[0] / 2
        # )
        et_dev = et.to(torch_device)

        print("running sim annealing on", fname)
        scores, rotamer_assignments = run_simulated_annealing(et_dev)

        scores_temp = scores
        scores = scores.cpu().numpy()
        numpy.set_printoptions(threshold=1e5)
        # print("scores", fname)
        # for i in range(scores.shape[1]):
        #     print(" ".join([str(val) for val in scores[:, i]]))

        scores = scores_temp[0, :]
        sort_scores, sort_inds = scores.sort()
        nkeep = min(scores.shape[0], 5)
        best_scores = sort_scores[0:nkeep]
        best_score_inds = sort_inds[0:nkeep]
        best_rot_assignments = rotamer_assignments[best_score_inds, :]

        scores = best_scores.cpu()

        rotamer_assignments = best_rot_assignments.cpu()
        print(
            "scores", " ".join([str(scores[i].item()) for i in range(scores.shape[0])])
        )

        validated_scores = torch.ops.tmol.validate_energies(
            et.nrotamers_for_res,
            et.oneb_offsets,
            et.res_for_rot,
            et.respair_nenergies,
            et.chunk_size,
            et.chunk_offset_offsets,
            et.twob_offsets,
            et.fine_chunk_offsets,
            et.energy1b,
            et.energy2b,
            rotamer_assignments,
        )

        # print("validated scores?", validated_scores)
        torch.testing.assert_allclose(scores, validated_scores)
