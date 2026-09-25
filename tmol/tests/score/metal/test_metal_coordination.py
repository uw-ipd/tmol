import os
from collections import Counter

import attr
import pytest
import torch
from yaml import safe_load

from tmol.database.scoring import MetalWellDepth
from tmol.io import atom_array_from_cif, pose_stack_from_biotite, pose_stack_from_pdb
from tmol.pack import PackerPalette, PackerTask, SetPackerTask
from tmol.pack.rotamer import build_rotamers
from tmol.score import ScoreType
from tmol.score.metal._metal_coordination_term import MetalCoordinationEnergyTerm
from tmol.tests.data import data_path
from tmol.tests.score.metal import metal_oracle

FIXTURE_DIR = data_path("metal_fixtures")

with open(os.path.join(FIXTURE_DIR, "expected.yaml")) as infile:
    EXPECTED = safe_load(infile)["fixtures"]

LOADABLE = [stem for stem, spec in EXPECTED.items() if "xfail" not in spec]


@pytest.fixture(scope="module")
def built():
    cache = {}

    def build(stem, device=torch.device("cpu")):
        if (stem, device) not in cache:
            structure = atom_array_from_cif(os.path.join(FIXTURE_DIR, stem + ".cif.gz"))
            cache[stem, device] = pose_stack_from_biotite(
                structure, device, prepare_ligands=True
            )
        return cache[stem, device]

    return build


def render(term, pose_stack, block_pair=False):
    for bt in pose_stack.packed_block_types.active_block_types:
        term.setup_block_type(bt)
    term.setup_packed_block_types(pose_stack.packed_block_types)
    term.setup_poses(pose_stack)
    if block_pair:
        return term.render_block_pair_scoring_module(pose_stack)
    return term.render_whole_pose_scoring_module(pose_stack)


def global_index(pose_stack, pose, block, atom):
    return int(pose_stack.block_coord_offset64[pose, block]) + atom


def contested_atoms(site_rows, fan_rows):
    """Atoms two restraints pull toward different places.

    A donor bridging two metals has an ideal position for each of them, and a
    cluster's own satisfier that is also another metal's donor has one for its
    cluster and one for that metal. No single coordinate answers both.
    """
    served = Counter((pose, block, atom) for pose, _, _, _, block, atom in site_rows)
    for pose, block, a, b in fan_rows:
        for atom in (a, b):
            if served[(pose, block, atom)]:
                served[(pose, block, atom)] += 1
    return {key for key, count in served.items() if count > 1}


def residuals(param_db, pose_stack, coords):
    """Energy an atom could have shed, energy it could not, and where it sits."""
    site_rows, site_params, fan_rows, fan_params = metal_oracle.restraints(
        param_db, pose_stack
    )
    contested = contested_atoms(site_rows, fan_rows)

    def at(pose, block, atom):
        return coords[pose, global_index(pose_stack, pose, block, atom)]

    def tensor(values):
        return torch.tensor(values, dtype=coords.dtype, device=coords.device)

    pbt = pose_stack.packed_block_types

    def label(pose, block, atom):
        bt = pbt.active_block_types[pose_stack.block_type_ind64[pose, block]]
        return f"{bt.name}:{bt.atoms[atom].name}"

    rows, settled, irreducible = [], 0.0, 0.0
    for (pose, mblock, matom, vatom, dblock, datom), params in zip(
        site_rows, site_params
    ):
        energy = float(
            metal_oracle.site_energies(
                at(pose, mblock, matom),
                at(pose, dblock, datom),
                at(pose, mblock, max(vatom, 0)),
                torch.tensor(vatom >= 0, device=coords.device),
                tensor(params),
            )
        )
        if (pose, dblock, datom) in contested:
            irreducible += energy
        else:
            settled += energy
        distance = float((at(pose, mblock, matom) - at(pose, dblock, datom)).norm())
        rows.append(
            (
                energy,
                f"site {label(pose, mblock, matom)}-{label(pose, dblock, datom)}"
                f" d={distance:.2f} d0={params[0]:.2f}",
            )
        )
    for (pose, block, a, b), params in zip(fan_rows, fan_params):
        energy = float(
            metal_oracle.fan_energies(
                at(pose, block, a), at(pose, block, b), tensor(params)
            )
        )
        if {(pose, block, a), (pose, block, b)} & contested:
            irreducible += energy
        else:
            settled += energy
        distance = float((at(pose, block, a) - at(pose, block, b)).norm())
        rows.append(
            (
                energy,
                f"fan {label(pose, block, a)}-{label(pose, block, b)}"
                f" d={distance:.2f} l0={params[0]:.2f}",
            )
        )
    rows.sort(reverse=True)
    worst = "; ".join(f"{text} -> {energy:.3f}" for energy, text in rows[:5])
    detail = f"settled {settled:.3f}, irreducible {irreducible:.3f}; worst {worst}"
    return settled, irreducible, detail


def idealized(param_db, pose_stack):
    """Coordinates with every paired donor moved onto its site at ideal distance.

    A cluster is first replaced by its ideal geometry, fitted onto it with a
    reflection allowed, since its atoms may be named in the mirror sense.
    """
    coords = pose_stack.coords.detach().clone().double()
    pbt = pose_stack.packed_block_types
    for pose, block in torch.nonzero(pose_stack.block_type_ind64 >= 0).tolist():
        bt = pbt.active_block_types[pose_stack.block_type_ind64[pose, block]]
        if not any(site.internal_satisfiers for site in bt.metal_sites):
            continue
        start = int(pose_stack.block_coord_offset64[pose, block])
        ideal = torch.tensor(
            [bt.ideal_coords[bt.icoors_index[a.name]] for a in bt.atoms],
            dtype=coords.dtype,
        )
        real = [j for j in range(bt.n_atoms) if not bt.atoms[j].name.startswith("V")]
        a, b = ideal[real], coords[pose, start + torch.tensor(real)].cpu()
        ca, cb = a.mean(0), b.mean(0)
        u, _, vt = torch.linalg.svd((a - ca).T @ (b - cb))
        placed = (ideal - ca) @ (u @ vt) + cb
        coords[pose, start : start + bt.n_atoms] = placed.to(coords.device)
    site_rows, site_params, _, _ = metal_oracle.restraints(param_db, pose_stack)
    targets = {}
    for (pose, mblock, matom, vatom, dblock, datom), params in zip(
        site_rows, site_params
    ):
        metal = coords[pose, global_index(pose_stack, pose, mblock, matom)]
        donor = global_index(pose_stack, pose, dblock, datom)
        if vatom >= 0:
            ray = coords[pose, global_index(pose_stack, pose, mblock, vatom)] - metal
        else:
            ray = coords[pose, donor] - metal
        targets.setdefault((pose, donor), []).append(
            metal + params[0] * ray / ray.norm()
        )
    # a donor bridging two metals carries one target per metal and cannot sit
    #    on both; its restraints balance at their mean
    for (pose, donor), points in targets.items():
        coords[pose, donor] = torch.stack(points).mean(0)
    return coords, site_rows


def with_well_depth(param_db, atom_type, donor, depth):
    metal_db = attr.evolve(
        param_db.scoring.metal_coordination,
        well_depths=(MetalWellDepth(atom_type=atom_type, donor=donor, depth=depth),),
    )
    scoring = attr.evolve(param_db.scoring, metal_coordination=metal_db)
    return attr.evolve(param_db, scoring=scoring)


@pytest.mark.parametrize("stem", LOADABLE)
def test_site_pairing_matches_detection(built, stem, default_database):
    pose_stack = built(stem)
    site_rows, _, _, _ = metal_oracle.restraints(default_database, pose_stack)
    info = pose_stack.pdb_info
    pbt = pose_stack.packed_block_types

    def key(pose, block):
        return (
            str(info.chain_labels[pose, block]),
            int(info.residue_labels[pose, block]),
        )

    found = {}
    for pose, mblock, _, _, dblock, datom in site_rows:
        bt = pbt.active_block_types[int(pose_stack.block_type_ind[pose, dblock])]
        found.setdefault(key(pose, mblock), set()).add(
            (*key(pose, dblock), bt.atoms[datom].name)
        )
    for metal in EXPECTED[stem]["metals"]:
        expected = {
            (d["chain"], d["res"], d["atom"])
            for d in EXPECTED[stem]["donors"][
                f"{metal['chain']}/{metal['comp']}/{metal['res']}"
            ]
        }
        assert found.get((metal["chain"], metal["res"]), set()) == expected


@pytest.mark.parametrize("stem", LOADABLE)
def test_kernel_matches_oracle(built, stem, default_database, torch_device):
    pose_stack = built(stem, torch_device)
    term = MetalCoordinationEnergyTerm(default_database, torch_device)
    coords, _ = idealized(default_database, pose_stack)
    generator = torch.Generator().manual_seed(0)
    noise = 0.1 * torch.randn(coords.shape, generator=generator, dtype=coords.dtype)
    coords = (coords + noise.to(coords.device)).requires_grad_(True)

    def gradient(energy):
        """Zero where a structure carries no restraint at all, as cobalt hexammine does."""
        (grad,) = torch.autograd.grad(energy.sum(), coords, allow_unused=True)
        return torch.zeros_like(coords) if grad is None else grad

    expected = metal_oracle.block_pair_energies(default_database, pose_stack, coords)
    expected_grad = gradient(expected)

    whole = render(term, pose_stack)(coords)
    torch.testing.assert_close(whole[0], expected.sum(dim=(1, 2)))
    torch.testing.assert_close(gradient(whole), expected_grad)

    pairs = render(term, pose_stack, block_pair=True)(coords)
    torch.testing.assert_close(pairs[0], expected)
    torch.testing.assert_close(gradient(pairs), expected_grad)


@pytest.mark.parametrize("stem", LOADABLE)
def test_ideal_sites_score_zero(built, stem, default_database, torch_device):
    """Every restraint one atom can satisfy on its own is at rest.

    An atom two restraints pull apart -- a cysteine bridging two metals, a
    fluoride that is one metal's satisfier and another's donor -- has an ideal
    position for each and can take only one, so what is left of those is the
    strain the ideal geometry cannot remove.
    """
    pose_stack = built(stem, torch_device)
    term = MetalCoordinationEnergyTerm(default_database, torch_device)
    coords, _ = idealized(default_database, pose_stack)

    settled, irreducible, detail = residuals(default_database, pose_stack, coords)
    assert settled < 1e-6, f"{stem}: {detail}"

    score = float(render(term, pose_stack)(coords).sum())
    assert score == pytest.approx(settled + irreducible, abs=1e-6), f"{stem}: {detail}"


def test_known_distortions(built, default_database):
    pose_stack = built("zn_tetrahedral_3ks3")
    term = MetalCoordinationEnergyTerm(default_database, torch.device("cpu"))
    scorer = render(term, pose_stack)
    coords, site_rows = idealized(default_database, pose_stack)
    sd_radial, sd_lateral = default_database.scoring.metal_coordination.widths("Zn2p")

    pose, mblock, matom, _, dblock, datom = site_rows[0]
    metal = coords[pose, global_index(pose_stack, pose, mblock, matom)]
    donor = global_index(pose_stack, pose, dblock, datom)
    u = (coords[pose, donor] - metal) / (coords[pose, donor] - metal).norm()
    perp = torch.linalg.cross(u, torch.tensor([1.0, 0.0, 0.0], dtype=u.dtype))
    perp = perp / perp.norm()

    radial = coords.clone()
    radial[pose, donor] += 0.05 * u
    assert float(scorer(radial).sum()) == pytest.approx(
        (0.05 / sd_radial) ** 2, rel=1e-5
    )

    # sideways by s leaves the distance off by sqrt(d0^2 + s^2) - d0
    lateral = coords.clone()
    lateral[pose, donor] += 0.1 * perp
    d0 = float((coords[pose, donor] - metal).norm())
    stretch = (d0**2 + 0.01) ** 0.5 - d0
    expected = (0.1 / sd_lateral) ** 2 + (stretch / sd_radial) ** 2
    assert float(scorer(lateral).sum()) == pytest.approx(expected, rel=1e-5)


def test_well_depth_counts_once_per_site(built, default_database):
    pose_stack = built("zn_tetrahedral_3ks3")
    param_db = with_well_depth(default_database, "Zn2p", "N", -5.0)
    term = MetalCoordinationEnergyTerm(param_db, torch.device("cpu"))
    site_rows, site_params, _, _ = metal_oracle.restraints(param_db, pose_stack)
    n_nitrogen = sum(1 for params in site_params if params[1] != 0)
    assert n_nitrogen > 0
    coords, _ = idealized(param_db, pose_stack)
    score = float(render(term, pose_stack)(coords).sum())
    assert score == pytest.approx(-5.0 * n_nitrogen, abs=1e-5)


def test_built_fan_is_at_rest(built, default_database):
    pose_stack = built("cu_zn_sod_3f7l")
    _, _, fan_rows, fan_params = metal_oracle.restraints(default_database, pose_stack)
    assert len(fan_rows) > 0
    coords = pose_stack.coords.double()
    for (pose, block, a, b), (l0, _) in zip(fan_rows, fan_params):
        xyz_a = coords[pose, global_index(pose_stack, pose, block, a)]
        xyz_b = coords[pose, global_index(pose_stack, pose, block, b)]
        assert float((xyz_a - xyz_b).norm()) == pytest.approx(l0, abs=1e-4)


def test_oracle_gradcheck():
    torch.manual_seed(0)
    n = 5
    metal = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
    donor = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
    virt = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
    has_virt = torch.tensor([True, True, False, True, False])
    site_params = torch.tensor([[2.1, -1.0, 0.1, 0.25]] * n, dtype=torch.float64)
    fan_params = torch.tensor([[2.1, 0.05]] * n, dtype=torch.float64)
    torch.autograd.gradcheck(
        lambda m, d, v: metal_oracle.site_energies(m, d, v, has_virt, site_params),
        (metal, donor, virt),
    )
    torch.autograd.gradcheck(
        lambda a, b: metal_oracle.fan_energies(a, b, fan_params), (metal, donor)
    )


def test_rotamer_scores_match_oracle(
    built, default_database, dun_sampler, torch_device
):
    """Every metal-rotamer x donor-rotamer pair, and each metal's fan."""
    pose_stack = built("zn_tetrahedral_3ks3", torch_device)
    task = PackerTask(pose_stack, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(dun_sampler)
    task = SetPackerTask.from_packer_task(task)
    pose_stack, rotamer_set = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    term = MetalCoordinationEnergyTerm(default_database, torch_device)
    render(term, pose_stack)
    scorer = term.render_rotamer_scoring_module(pose_stack, rotamer_set)
    energies = scorer.forward_split(rotamer_set.coords).coalesce()
    _, pose_ind, rot1, rot2 = energies.indices().cpu()
    got = {
        (int(p), int(a), int(b)): float(v)
        for p, a, b, v in zip(pose_ind, rot1, rot2, energies.values().cpu())
    }

    site_rows, site_params, fan_rows, fan_params = metal_oracle.restraints(
        default_database, pose_stack
    )
    coords = rotamer_set.coords.detach().double().cpu()
    offset = rotamer_set.coord_offset_for_rot.cpu()
    rot_offset = rotamer_set.rot_offset_for_block.cpu()
    n_rots = rotamer_set.n_rots_for_block.cpu()
    bt_for_rot = rotamer_set.block_type_ind_for_rot.cpu()

    def rots(pose, block):
        start = int(rot_offset[pose, block])
        return range(start, start + int(n_rots[pose, block]))

    expected = {}
    donor_rot_counts = []
    for (pose, mblock, matom, vatom, dblock, datom), params in zip(
        site_rows, site_params
    ):
        donor_rots = rots(pose, dblock)
        donor_rot_counts.append(len(donor_rots))
        for rm in rots(pose, mblock):
            for rd in donor_rots:
                assert bt_for_rot[rd] == pose_stack.block_type_ind64[pose, dblock]
                e = metal_oracle.site_energies(
                    coords[offset[rm] + matom],
                    coords[offset[rd] + datom],
                    coords[offset[rm] + max(vatom, 0)],
                    torch.tensor(vatom >= 0),
                    torch.tensor(params, dtype=torch.float64),
                )
                key = (pose, min(rm, rd), max(rm, rd))
                expected[key] = expected.get(key, 0.0) + float(e)
    for (pose, block, a, b), params in zip(fan_rows, fan_params):
        for rm in rots(pose, block):
            e = metal_oracle.fan_energies(
                coords[offset[rm] + a],
                coords[offset[rm] + b],
                torch.tensor(params, dtype=torch.float64),
            )
            key = (pose, rm, rm)
            expected[key] = expected.get(key, 0.0) + float(e)

    assert max(donor_rot_counts) > 1, "no metal donor was given rotamers"
    assert got.keys() == expected.keys()
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-4, abs=1e-3), key


def test_block_pair_sums_to_whole_pose(built, default_database, torch_device):
    pose_stack = built("mg_one_donor_4e3y", torch_device)
    term = MetalCoordinationEnergyTerm(default_database, torch_device)
    coords, site_rows = idealized(default_database, pose_stack)
    pose, mblock, matom = site_rows[0][:3]
    coords[pose, global_index(pose_stack, pose, mblock, matom)] += 0.3
    whole = render(term, pose_stack)(coords)
    pairs = render(term, pose_stack, block_pair=True)(coords)
    torch.testing.assert_close(pairs.sum(dim=(2, 3)), whole)


def test_pose_without_metals_is_zero(ubq_pdb, default_database, torch_device):
    pose_stack = pose_stack_from_pdb(ubq_pdb, torch_device)
    term = MetalCoordinationEnergyTerm(default_database, torch_device)
    render(term, pose_stack)
    assert term.pose_score_term_is_invariant_zero(pose_stack)


def test_beta2016_includes_metal_coordination(torch_device):
    from tmol.score import beta2016_score_function

    sfxn = beta2016_score_function(torch_device)
    assert float(sfxn.get_weight(ScoreType.metal_coordination)) == 1.0
