import zarr
import numpy
import torch
import attr

from tmol.pack.datatypes import PackerEnergyTables
from tmol.pack.simulated_annealing import run_simulated_annealing
from tmol.utility.cumsum import exclusive_cumsum


def load_ig_from_file(fname):
    store = zarr.ZipStore(fname)
    zgroup = zarr.group(store=store)
    nres = zgroup.attrs["nres"]
    oneb_energies = {}
    twob_energies = {}
    for i in range(1,nres+1):
        oneb_arrname = "%d" % i
        oneb_energies[oneb_arrname] = numpy.array(zgroup[oneb_arrname], dtype=float)
        for j in range(i+1,nres+1):
            twob_arrname = "%d-%d" % (i, j)
            if twob_arrname in zgroup:
                twob_energies[twob_arrname] = numpy.array(zgroup[twob_arrname], dtype=float)
    return oneb_energies, twob_energies

def test_load_ig():
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    assert len(oneb) == 76
    nrots = numpy.zeros((76,), dtype=int)
    for i in range(76):
        arrname = "%d" % (i+1)
        nrots[i] = oneb[arrname].shape[0]
    for i in range(76):
        for j in range(i+1, 76):
            arrname = "%d-%d" % (i+1, j+1)
            if arrname in twob:
                assert nrots[i] == twob[arrname].shape[0]
                assert nrots[j] == twob[arrname].shape[1]

def count_table_size(twob):
    count = 0
    for tabname in twob:
        shape = twob[tabname].shape
        count += shape[0] * shape[1]
    return count

def create_twobody_energy_table(oneb, twob):
    nres = len(oneb)
    offsets = numpy.zeros((nres, nres), dtype=numpy.int64)
    nenergies = numpy.zeros((nres, nres), dtype=int)
    nrotamers_for_res = numpy.array([oneb["{}".format(i+1)].shape[0] for i in range(nres)], dtype=int)
    nrots_total = numpy.sum(nrotamers_for_res)
    oneb_offsets = exclusive_cumsum(nrotamers_for_res)

    energy1b = numpy.zeros(nrots_total, dtype=float)
    res_for_rot = numpy.zeros(nrots_total, dtype=int)
    for i in range(nres):
        tablename = "{}".format(i+1)
        table = oneb[tablename]
        start = oneb_offsets[i]
        energy1b[(start):(start+table.shape[0])] = table
        res_for_rot[(start):(start+table.shape[0])] = i

    for i in range(nres):
        for j in range(i+1,nres):
            tabname = "{}-{}".format(i+1,j+1)
            if tabname in twob:
                nenergies[i,j] = nrotamers_for_res[i]*nrotamers_for_res[j]
                nenergies[j,i] = nrotamers_for_res[i]*nrotamers_for_res[j]

    twob_offsets = exclusive_cumsum(nenergies.reshape(-1)).reshape(nenergies.shape)
    n_rpes_total = numpy.sum(nenergies)
    energy2b = numpy.zeros(n_rpes_total, dtype=float)
    for i in range(nres):
        for j in range(i+1, nres):
            if nenergies[i,j] == 0:
                continue
            tabname = "{}-{}".format(i+1,j+1)
            table = twob[tabname]
            start_ij = twob_offsets[i,j]
            extent = nenergies[i,j]
            energy2b[start_ij:(start_ij+extent)] = table.reshape(-1)
            start_ji = twob_offsets[j,i]
            energy2b[start_ji:(start_ji+extent)] = table.T.reshape(-1)

    return PackerEnergyTables(
        nrotamers_for_res=nrotamers_for_res,
        oneb_offsets=oneb_offsets,
        res_for_rot=res_for_rot,
        nenergies=nenergies,
        twob_offsets=twob_offsets,
        energy1b=energy1b,
        energy2b=energy2b
    )

def test_energy_table_construction():
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    energy_tables = create_twobody_energy_table(oneb, twob)
    et = energy_tables

    nrots_total = et.res_for_rot.shape[0]
    # pick two residues, 12 and 14
    assert "12-14" in twob

    for i in range(et.oneb_offsets[11],et.oneb_offsets[11]+et.nrotamers_for_res[11]):
        ires = et.res_for_rot[i]
        assert ires == 11
        irot_on_res = i - et.oneb_offsets[ires]
        for j in range(et.oneb_offsets[13], et.oneb_offsets[13]+et.nrotamers_for_res[13]):
            jres = et.res_for_rot[j]
            assert jres == 13
            jrot_on_res = j - et.oneb_offsets[jres]
            if et.nenergies[ires, jres] == 0:
                continue
            ij_energy = et.energy2b[
                et.twob_offsets[ires,jres]
                + irot_on_res * et.nrotamers_for_res[jres]
                + jrot_on_res
            ]
            ji_energy = et.energy2b[
                et.twob_offsets[jres,ires]
                + jrot_on_res * et.nrotamers_for_res[ires]
                + irot_on_res
            ]
            assert ij_energy == ji_energy # exact equality ok since they are copies

def test_run_sim_annealing(torch_device):
    # torch_device = torch.device("cpu")

    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    et = create_twobody_energy_table(oneb, twob)

    print("nrotamers", et.res_for_rot.shape[0])
    et_dev = et.to(torch_device)

    scores, rotamer_assignments = run_simulated_annealing(et_dev)

    sort_scores, sort_inds = scores[0,:].sort()
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
    #print("assignment 0", rotamer_assignments[0,0:20])
    #print("sorted assignment 0", best_rot_assignments[0,0:20])

    validated_scores = torch.ops.tmol.validate_energies(
        et.nrotamers_for_res,
        et.oneb_offsets,
        et.res_for_rot,
        et.nenergies,
        et.twob_offsets,
        et.energy1b,
        et.energy2b,
        rotamer_assignments)

    print("validated scores?", validated_scores)
    torch.testing.assert_allclose(scores, validated_scores)


def test_run_sim_annealing_on_repacking_jobs():
    torch_device = torch.device("cuda")
    fnames = ["1wzbFHA", "1qtxFHB", "1kd8FHB", "1ojhFHA", "1ff4FHA", "1vmgFHA", "1u36FHA", "1w0nFHA"]
    # fnames = ["1u36FHA"]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs/repack/" + fname + "_repack.zarr"
        oneb, twob = load_ig_from_file(path_to_zarr_file)
        et = create_twobody_energy_table(oneb, twob)
        # print("nrotamers", et.res_for_rot.shape[0])
        # print("table size", count_table_size(twob))
        et_dev = et.to(torch_device)

        # print("running sim annealing on", fname)
        scores, rotamer_assignments = run_simulated_annealing(et_dev)

        scores_temp = scores
        scores = scores.cpu().numpy()
        numpy.set_printoptions(threshold=1e5)
        print("scores", fname)
        for i in range(scores.shape[1]):
            print(scores[0,i], scores[1, i])

        scores = scores_temp[1,:]
        sort_scores, sort_inds = scores.sort()
        nkeep = min(scores.shape[0], 5)
        best_scores = sort_scores[0:nkeep]
        best_score_inds = sort_inds[0:nkeep]
        best_rot_assignments = rotamer_assignments[best_score_inds, :]

        scores = best_scores.cpu()

        rotamer_assignments = best_rot_assignments.cpu()
        #print("scores", " ".join([str(scores[i].item()) for i in range(scores.shape[0])]))


        validated_scores = torch.ops.tmol.validate_energies(
            et.nrotamers_for_res,
            et.oneb_offsets,
            et.res_for_rot,
            et.nenergies,
            et.twob_offsets,
            et.energy1b,
            et.energy2b,
            rotamer_assignments)

        # print("validated scores?", validated_scores)
        torch.testing.assert_allclose(scores, validated_scores)

def test_run_sim_annealing_on_redes_ex1ex2_jobs():
    torch_device = torch.device("cuda")
    fnames = ["1wzbFHA", "1qtxFHB", "1kd8FHB", "1ojhFHA", "1ff4FHA", "1vmgFHA", "1u36FHA", "1w0nFHA"]
    # fnames = ["1w0nFHA"]
    # fnames = ["1u36FHA", "1w0nFHA"]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs/redes_ex1ex2/" + fname + "_redes_ex1ex2.zarr"
        oneb, twob = load_ig_from_file(path_to_zarr_file)
        #print("table size", count_table_size(twob))
        et = create_twobody_energy_table(oneb, twob)
        #print("energy2b", et.energy2b.shape[0])
        #print("nrotamers", et.res_for_rot.shape[0])
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

        scores = scores_temp[0,:]
        sort_scores, sort_inds = scores.sort()
        nkeep = min(scores.shape[0], 5)
        best_scores = sort_scores[0:nkeep]
        best_score_inds = sort_inds[0:nkeep]
        best_rot_assignments = rotamer_assignments[best_score_inds, :]

        scores = best_scores.cpu()

        rotamer_assignments = best_rot_assignments.cpu()
        print("scores", " ".join([str(scores[i].item()) for i in range(scores.shape[0])]))


        validated_scores = torch.ops.tmol.validate_energies(
            et.nrotamers_for_res,
            et.oneb_offsets,
            et.res_for_rot,
            et.nenergies,
            et.twob_offsets,
            et.energy1b,
            et.energy2b,
            rotamer_assignments)

        # print("validated scores?", validated_scores)
        torch.testing.assert_allclose(scores, validated_scores)
