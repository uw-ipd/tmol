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

def create_twobody_energy_table(oneb, twob):
    nres = len(oneb)
    offsets = numpy.zeros((nres, nres), dtype=int)
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
    fname = "1ubq_ig"
    oneb, twob = load_ig_from_file(fname)
    et = create_twobody_energy_table(oneb, twob)
    et = et.to(torch_device)

    scores, rotamer_assignments = run_simulated_annealing(et)
    scores = scores.cpu()
    print("scores", scores)
    
