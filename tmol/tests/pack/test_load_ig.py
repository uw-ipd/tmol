import zarr
import numpy
import torch
import attr

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args

from tmol.pack.datatypes import PackerEnergyTables
from tmol.pack.simulated_annealing import (
    run_one_stage_simulated_annealing,
    run_multi_stage_simulated_annealing
)
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
    nrotamers_for_res = numpy.array([oneb["{}".format(i+1)].shape[-1] for i in range(nres)], dtype=int)
    nrots_total = numpy.sum(nrotamers_for_res)
    oneb_offsets = exclusive_cumsum(nrotamers_for_res)

    for key in oneb:
        example = oneb[key]
        if len(example.shape) == 1:
            energy1b = numpy.zeros((1,nrots_total), dtype=float)
        else:
            energy1b = numpy.zeros((example.shape[0], nrots_total), dtype=float)
        break


    res_for_rot = numpy.zeros(nrots_total, dtype=int)
    for i in range(nres):
        tablename = "{}".format(i+1)
        table = oneb[tablename]
        start = oneb_offsets[i]
        if len(table.shape) == 1:
            energy1b[0, (start):(start+table.shape[0])] = table
            res_for_rot[(start):(start+table.shape[0])] = i
        else:
            energy1b[:, (start):(start+table.shape[1])] = table
            res_for_rot[(start):(start+table.shape[1])] = i

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

@validate_args
def energy_from_state_assignment(
    ig: PackerEnergyTables,
    assignments: Tensor(torch.int32)[:,:],
    bg_assignment=None
) -> Tensor(torch.float)[:] :
    if bg_assignment is None:
        n_assignments = assignments.shape[0]
        bg_assignment = torch.zeros((n_assignments,), dtype=torch.int32)
    scores = torch.ops.tmol.validate_energies(
        ig.nrotamers_for_res,
        ig.oneb_offsets,
        ig.res_for_rot,
        ig.nenergies,
        ig.twob_offsets,
        ig.energy1b,
        ig.energy2b,
        assignments,
        bg_assignment
    )
    return scores


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

def default_simA_params():
    params = torch.zeros((1,4), dtype=torch.float)
    params[0,0] = 30
    params[0,1] = 0.3
    params[0,2] = 10
    params[0,3] = 1/8
    return params
            
def test_run_sim_annealing(torch_device):
    # torch_device = torch.device("cpu")

    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    et = create_twobody_energy_table(oneb, twob)

    print("nrotamers", et.res_for_rot.shape[0])
    et_dev = et.to(torch_device)

    params = default_simA_params()
    scores, rotamer_assignments, bg_inds = run_one_stage_simulated_annealing(params, et_dev)

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

    bg_inds = bg_inds.cpu()
    validated_scores = torch.ops.tmol.validate_energies(
        et.nrotamers_for_res,
        et.oneb_offsets,
        et.res_for_rot,
        et.nenergies,
        et.twob_offsets,
        et.energy1b,
        et.energy2b,
        rotamer_assignments,
        bg_inds
    )

    print("validated scores?", validated_scores)
    torch.testing.assert_allclose(scores, validated_scores)


def test_run_sim_annealing_on_repacking_jobs():
    torch.manual_seed(1)
    
    torch_device = torch.device("cuda")
    fnames = ["1wzbFHA", "1qtxFHB", "1kd8FHB", "1ojhFHA", "1ff4FHA", "1vmgFHA", "1u36FHA", "1w0nFHA"]
    # fnames = ["1u36FHA"]
    simA_params = default_simA_params()
    for fname in fnames:
        path_to_zarr_file = "zarr_igs/repack/" + fname + "_repack.zarr"
        oneb, twob = load_ig_from_file(path_to_zarr_file)
        et = create_twobody_energy_table(oneb, twob)
        # print("nrotamers", et.res_for_rot.shape[0])
        # print("table size", count_table_size(twob))
        et_dev = et.to(torch_device)

        # print("running sim annealing on", fname)
        scores, rotamer_assignments, bg_inds = run_one_stage_simulated_annealing(simA_params, et_dev)
        bg_inds = bg_inds.cpu()

        # scores_temp = scores
        # scores = scores.cpu().numpy()
        # numpy.set_printoptions(threshold=1e5)
        # print("scores", fname)
        # for i in range(scores.shape[1]):
        #     print(scores[0,i], scores[1, i])
        #scores = scores_temp[1,:]

        scores = scores[0,:]
        sort_scores, sort_inds = scores.sort()
        nkeep = min(scores.shape[0], 5)
        best_scores = sort_scores[0:nkeep]
        best_score_inds = sort_inds[0:nkeep]
        best_rot_assignments = rotamer_assignments[best_score_inds, :]

        print("scores", scores)
        print("sort_scores", sort_scores)
        print("best_score_inds", best_score_inds)
        print("bg_inds", bg_inds)
        
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
            rotamer_assignments,
            bg_inds
        )

        # print("validated scores?", validated_scores)
        torch.testing.assert_allclose(scores, validated_scores)

def test_run_sim_annealing_on_redes_ex1ex2_jobs():
    torch_device = torch.device("cuda")
    fnames = ["1wzbFHA", "1qtxFHB", "1kd8FHB", "1ojhFHA", "1ff4FHA", "1vmgFHA", "1u36FHA", "1w0nFHA"]
    # fnames = ["1w0nFHA"]
    # fnames = ["1u36FHA", "1w0nFHA"]
    simA_params = default_simA_params()
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
        scores, rotamer_assignments, bg_inds = run_one_stage_simulated_annealing(simA_params, et_dev)
        bg_inds = bg_inds.cpu()

        scores_temp = scores
        #scores = scores.cpu().numpy()
        #numpy.set_printoptions(threshold=1e5)
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


        background_inds = torch.zeros((1,), dtype=torch.int32)
        validated_scores = torch.ops.tmol.validate_energies(
            et.nrotamers_for_res,
            et.oneb_offsets,
            et.res_for_rot,
            et.nenergies,
            et.twob_offsets,
            et.energy1b,
            et.energy2b,
            rotamer_assignments,
            bg_inds
        )

        # print("validated scores?", validated_scores)
        torch.testing.assert_allclose(scores, validated_scores)

def create_residue_subsamples(nres, subset_size, rot_limit, neighbors, oneb):
    selected = numpy.full((nres,), 0, dtype=int)
    resorder = numpy.random.permutation(nres)
    subsets = []
    for i in range(nres):
        ires = resorder[i]
        if selected[ires] == 1:
            continue
        i_count_nrots = oneb["{}".format(i+1)].shape[0]
        neighbs = neighbors[ires]
        neighbs_up_to_limit = None
        if neighbs.shape[0] <= subset_size:
            neighbs_up_to_limit = neighbs
        else:
            neighbs_up_to_limit = neighbs[:subset_size]

        selected_neighbs = []
        for neighb in neighbs_up_to_limit:
            neighb_nrots = oneb["{}".format(neighb+1)].shape[0]
            if i_count_nrots + neighb_nrots > rot_limit:
                #print("skipping residue", neighb, "with", neighb_nrots, "rots", i_count_nrots, "nearing", rot_limit)
                continue
            selected[neighb] = 1
            selected_neighbs.append(neighb)
            i_count_nrots += neighb_nrots

        subsets.append(numpy.sort(numpy.array(selected_neighbs, dtype=int)))
        #print("subset", i_count_nrots)
    return subsets

def create_res_subset_ig(full_oneb, full_twob, subset, state_assignments):
    assert len(state_assignments.shape) == 2
    oneb_subset = {}
    twob_subset = {}
    n_backgrounds = state_assignments.shape[0]

    nres = len(full_oneb)
    res_in_subset = numpy.full((nres,), 0, dtype=int)
    for res in subset:
        res_in_subset[res] = 1
    
    for subset_count, res in enumerate(subset):
        res_oneb = numpy.copy(full_oneb["{}".format(res+1)])
        res_oneb = numpy.tile(res_oneb, n_backgrounds).reshape(n_backgrounds, -1)
        
        for i in range(nres):
            if res == i:
                continue
            if res_in_subset[i] == 1:
                continue
            lower = res if res < i else i
            upper = res if res > i else i
            edge_name = "{}-{}".format(lower+1,upper+1)
            if edge_name not in full_twob:
                continue
            edge_table = full_twob[edge_name]

            i_state = state_assignments[:, i]
            if res < i:
                res_oneb += edge_table[:, i_state].transpose()
            else:
                res_oneb += edge_table[i_state, :]
        oneb_subset["{}".format(subset_count+1)] = res_oneb

    for i, res1 in enumerate(subset):
        for j, res2 in enumerate(subset):
            if j <= i:
                continue
            edge_name = "{}-{}".format(res1+1, res2+1)
            if edge_name not in full_twob:
                continue
            new_edge_name = "{}-{}".format(i+1, j+1)
            twob_subset[new_edge_name] = full_twob[edge_name]
    return oneb_subset, twob_subset

def neighbors_from_ig(nres, twob):
    neighbors = []
    for i in range(nres):
        i_neighbs = []
        for j in range(nres):
            if i == j:
                continue
            lower = i if i < j else j
            upper = j if i < j else i
            edge_name = "{}-{}".format(lower+1,upper+1)
            if edge_name in twob:
                i_neighbs.append(j)
        neighbors.append(numpy.array(i_neighbs, dtype=int))
    return neighbors

def ranked_neighbors_from_ig(nres, twob):
    neighbors = []
    for i in range(nres):
        i_neighbs = []
        i_neighb_rank = []
        for j in range(nres):
            if i == j:
                continue
            lower = i if i < j else j
            upper = j if i < j else i
            edge_name = "{}-{}".format(lower+1,upper+1)
            if edge_name in twob:
                i_neighbs.append(j)
                ij_edge = twob[edge_name]
                n_nonzero = numpy.nonzero(ij_edge)[0].shape[0]
                i_neighb_rank.append(n_nonzero)
        i_neighbs = numpy.array(i_neighbs, dtype=int)
        i_neighb_rank = numpy.array(i_neighb_rank, dtype=int)
        ranked = numpy.flip(numpy.argsort(i_neighb_rank), axis=0)
        i_neighbs = i_neighbs[ranked]
        i_neighbs = numpy.concatenate((numpy.array([i], dtype=int), i_neighbs))
        # print("ranked neigbs", i, i_neighbs)
        neighbors.append(i_neighbs)
    return neighbors

def test_rank_neighbors():
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    nres = len(oneb)
    neighbors = ranked_neighbors_from_ig(nres, twob)
    res36_neighbs = neighbors[36]
    rank_of_neighbors = []
    for neighb in res36_neighbs:
        if neighb == 36:
            continue
        lower = 36 if neighb > 36 else neighb
        upper = 36 if neighb < 36 else neighb
        edge_name = "{}-{}".format(lower+1, upper+1)
        assert edge_name in twob
        n_nonzero = numpy.nonzero(twob[edge_name])[0].shape[0]
        rank_of_neighbors.append(n_nonzero)
    for i in range(len(rank_of_neighbors)-1):
        assert rank_of_neighbors[i] >= rank_of_neighbors[i+1]
        

def test_create_residue_subsamples():
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    nres = len(oneb)
    neighbors = ranked_neighbors_from_ig(nres, twob)

    subset_size = 30
    subsets = create_residue_subsamples(nres, subset_size, 4000, neighbors, oneb)

    #for subset in subsets:
    #    print(subset)

def random_assignment(oneb):
    nres = len(oneb)
    assignment = numpy.zeros((nres,), dtype=int)
    for i in range(nres):
        nrots = oneb["{}".format(i+1)].shape[0]
        assignment[i] = numpy.random.randint(nrots)
    return assignment

def random_assignments(oneb, nassignments):
    nres = len(oneb)
    assignments = numpy.zeros((nassignments, nres), dtype=int)
    for i in range(nassignments):
        assignments[i, :] = random_assignment(oneb)
    return assignments

def test_create_subsample_ig():
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    nres = len(oneb)
    neighbors = neighbors_from_ig(nres, twob)

    chunk_size = 16
    subset_size = 30
    subsets = create_residue_subsamples(nres, subset_size, 20000, neighbors, oneb)
    subset0 = subsets[0]

    num_assignments = 2
    
    faux_assignments = random_assignments(oneb, num_assignments)
    oneb_subset, twob_subset = create_res_subset_ig(oneb, twob, subset0, faux_assignments)

    subset_assignment1 = torch.tensor(random_assignment(oneb_subset)[None, :], dtype=torch.int32)
    subset_assignment2 = torch.tensor(random_assignment(oneb_subset)[None, :], dtype=torch.int32)
 
    full_ig = create_twobody_energy_table(oneb, twob)
    subset_ig = create_twobody_energy_table(oneb_subset, twob_subset)

    for i in range(num_assignments):
        full_assignment1 = torch.tensor(numpy.copy(faux_assignments[i])[None, :], dtype=torch.int32)
        full_assignment2 = torch.tensor(numpy.copy(faux_assignments[i])[None, :], dtype=torch.int32)
        full_assignment1[0, subset0] = subset_assignment1
        full_assignment2[0, subset0] = subset_assignment2
        # for i, res in enumerate(subset0):
        #     full_assignment1[res] = subset_assignment1[i]
        #     full_assignment2[res] = subset_assignment2[i]

        bg_assignment = torch.full((1,), i, dtype=torch.int32)
        
        full_energy1 = energy_from_state_assignment(full_ig, full_assignment1)
        full_energy2 = energy_from_state_assignment(full_ig, full_assignment2)
        subset_energy1 = energy_from_state_assignment(subset_ig, subset_assignment1, bg_assignment)
        subset_energy2 = energy_from_state_assignment(subset_ig, subset_assignment2, bg_assignment)
    
        full_deltaE = full_energy1 - full_energy2
        subset_deltaE = subset_energy1 - subset_energy2
    
        # print(full_energy1, full_energy2, full_deltaE)
        # print(subset_energy1, subset_energy2, subset_deltaE)
        assert abs(full_deltaE - subset_deltaE) < 1e-2

def pack_neighborhoods(oneb, twob, torch_device):
    full_ig = create_twobody_energy_table(oneb, twob)
    nres = len(oneb)
    neighbors = ranked_neighbors_from_ig(nres, twob)
    n_backgrounds = 100

    full_assignments = torch.tensor(
        random_assignments(oneb, n_backgrounds), dtype=torch.int32
    )

    subset_size = 20
    rotamer_limit = 15000
    n_repeats = 6
    count = 0
    simA_params = default_simA_params()
    for repeat in range(n_repeats):
        subsets = create_residue_subsamples(nres, subset_size, rotamer_limit, neighbors, oneb)
            
        for subset in subsets:
            count += 1
            oneb_subset, twob_subset = create_res_subset_ig(oneb, twob, subset, full_assignments)
            ig = create_twobody_energy_table(oneb_subset, twob_subset)
            print( "mem", (4 * (ig.energy2b.shape[0] + ig.energy1b.shape[0] * ig.energy1b.shape[1])) // (1028*1028) )
            ig_dev = ig.to(torch_device)
            
            start_assignments = full_assignments[:, subset]
            start_scores = energy_from_state_assignment(
                ig, start_assignments,
                torch.arange(n_backgrounds, dtype=torch.int32)
            )
            
            scores, rot_assignments, background_inds = run_multi_stage_simulated_annealing(simA_params, ig_dev)

            best_subset_assignments = rot_assignments[0:n_backgrounds].cpu()
            best_background_assignments = background_inds[0:n_backgrounds].cpu()
            best_background_assignments64 = best_background_assignments.to(torch.int64)
            full_assignments = full_assignments[best_background_assignments64,:]
            full_assignments[:,subset] = best_subset_assignments
            print("n_unique", full_assignments.unique(dim=0).shape[0])
            # end_score = energy_from_state_assignment(ig, best_subset_assignments, best_background_assignments)

            validated_scores = energy_from_state_assignment(full_ig, full_assignments)

            #print(start_score.item(), end_score.item(), (end_score - start_score).item())
            #print(last_score.item(), validated_scores[0].item(), (validated_scores[0] - last_score).item() )
            print(repeat, count, "validated scores", validated_scores[0:3])
            # print("after pack #", count, validated_scores[0].item())
            # last_score = validated_scores[0]
            # print("packed", subset.shape[0], "residues, scores", scores[0], validated_scores[0])
    validated_scores = energy_from_state_assignment(full_ig, full_assignments)
    return validated_scores[0], full_assignments[0]
        
def test_pack_subsamples():
    torch_device = torch.device("cuda")
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    pack_neighborhoods(oneb, twob, torch_device)

def test_run_pack_neighborhoods_on_redes_ex1ex2_jobs():
    torch_device = torch.device("cuda")
    fnames = ["1wzbFHA", "1qtxFHB", "1kd8FHB", "1ojhFHA", "1ff4FHA", "1vmgFHA", "1u36FHA", "1w0nFHA"]
    # fnames = ["1w0nFHA"]
    # fnames = ["1u36FHA", "1w0nFHA"]
    for fname in fnames:
        path_to_zarr_file = "zarr_igs/redes_ex1ex2/" + fname + "_redes_ex1ex2.zarr"
        oneb, twob = load_ig_from_file(path_to_zarr_file)
        print("packing", fname)
        score, assignment = pack_neighborhoods(oneb, twob, torch_device)
        print("final score", score)
