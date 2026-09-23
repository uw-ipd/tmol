"""Metal coordination restraints, computed in plain torch."""

from itertools import combinations

import numpy
import torch

from tmol.chemical import RefinedResidueType
from tmol.database import ParameterDatabase
from tmol.database.chemical import ideal_distances, metal_table
from tmol.pose import PoseStack
from tmol.score.common._scoring_module import _coordinate_independent_score

from .._energy_term import EnergyTerm


class MetalCoordinationEnergyTerm(EnergyTerm):
    """Hold each donor on its site's vertex ray and each site fan rigid.

    Per occupied site: a harmonic on the metal-donor distance and on the
    donor's displacement off the ray from the metal through the site virtual.
    Untemplated ions keep only the distance. The fan term holds metal and
    virtual separations at ideal; it is zero whenever the fan is built from
    its icoors, and only acts in cartesian minimization.

    A site is occupied when its connection on the metal is filled; the donor
    is the atom on the partner's side of that connection.
    """

    # radial, lateral, fan widths in A
    widths = (0.1, 0.25, 0.05)

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super().__init__(param_db=param_db, device=device)
        self.device = device
        table = metal_table()
        self.ion_for_name3 = {ion["name3"]: ion for ion in table["ions"]}
        self.donor_radii = table["donor_radii"]
        self.vertices_for = {g["name"]: g["vertices"] for g in table["geometries"]}
        # water keeps its own distance key, as in detection
        self.distance_key = {
            at.name: ("Owat" if at.name == "Owat" else at.element)
            for at in param_db.chemical.atom_types
        }

    @classmethod
    def class_name(cls):
        return "MetalCoordination"

    @classmethod
    def score_types(cls):
        import tmol.score.terms._metal_coordination_creator

        return (
            tmol.score.terms._metal_coordination_creator.MetalCoordinationTermCreator.score_types()
        )

    def n_bodies(self):
        return 2

    def setup_block_type(self, block_type: RefinedResidueType):
        super().setup_block_type(block_type)

    def pose_score_term_is_invariant_zero(self, pose_stack: PoseStack):
        bts = pose_stack.packed_block_types.active_block_types
        used = torch.unique(pose_stack.block_type_ind64).tolist()
        return not any(bts[i].metal_sites for i in used if i >= 0)

    def get_pose_score_term_function(self):
        return metal_coordination_pose_scores

    def get_score_term_attributes(self, pose_stack: PoseStack):
        site_rows, site_d0, fan_rows, fan_l0 = self.restraints(pose_stack)
        device = pose_stack.device
        return [
            torch.tensor(site_rows, dtype=torch.int64, device=device).view(-1, 6),
            torch.tensor(site_d0, dtype=torch.float32, device=device),
            torch.tensor(fan_rows, dtype=torch.int64, device=device).view(-1, 4),
            torch.tensor(fan_l0, dtype=torch.float32, device=device),
            torch.tensor(self.widths, dtype=torch.float32, device=device),
        ]

    def restraints(self, pose_stack: PoseStack):
        """Site and fan restraint rows for every metal in the stack.

        Site rows are (pose, metal block, metal atom, virtual atom or -1, donor
        block, donor atom) with the ideal metal-donor distance; fan rows are
        (pose, block, atom, atom) with the ideal separation.
        """
        pbt = pose_stack.packed_block_types
        bt_inds = pose_stack.block_type_ind64.cpu().numpy()
        irc = pose_stack.inter_residue_connections64.cpu().numpy()

        site_rows, site_d0, fan_rows, fan_l0 = [], [], [], []
        for pose, block in zip(*numpy.nonzero(bt_inds >= 0)):
            bt = pbt.active_block_types[bt_inds[pose, block]]
            if not bt.metal_sites or bt.metal_sites[0].internal_satisfiers:
                continue
            site = bt.metal_sites[0]
            dist = ideal_distances(self.ion_for_name3[bt.name3], self.donor_radii)
            metal = bt.atom_to_idx[site.metal_atom]
            virts = [bt.atom_to_idx[v] for v in site.site_virts]

            for k, name in enumerate(site.site_connections):
                partner, partner_conn = irc[pose, block, bt.connection_to_cidx[name]]
                if partner < 0:
                    continue
                other = pbt.active_block_types[bt_inds[pose, partner]]
                donor = other.atom_to_idx[other.connections[partner_conn].atom]
                v = virts[k] if virts else -1
                site_rows.append((pose, block, metal, v, partner, donor))
                key = self.distance_key[other.atoms[donor].atom_type]
                site_d0.append(dist.get(key, dist["O"]))

            ideal = self.ideal_fan(bt, site) if virts else {}
            for a, b in combinations(list(ideal), 2):
                fan_rows.append((pose, block, a, b))
                fan_l0.append(numpy.linalg.norm(ideal[a] - ideal[b]))
        return site_rows, site_d0, fan_rows, fan_l0

    def ideal_fan(self, bt, site):
        """Ideal positions of the metal and its site virtuals, by atom index."""
        d = {ic.name: ic.d for ic in bt.icoors}
        verts = numpy.asarray(self.vertices_for[site.geometry], dtype=numpy.float64)
        verts /= numpy.linalg.norm(verts, axis=1, keepdims=True)
        out = {bt.atom_to_idx[site.metal_atom]: numpy.zeros(3)}
        for vertex, name in zip(verts, site.site_virts):
            out[bt.atom_to_idx[name]] = vertex * d[name]
        return out


def site_energies(metal, donor, virt, has_virt, d0, widths):
    """Radial and lateral energy per site; lateral is zero without a virtual."""
    delta = donor - metal
    r = torch.linalg.norm(delta, dim=-1)
    radial = ((r - d0) / widths[0]) ** 2
    ray = torch.where(has_virt.unsqueeze(-1), virt - metal, delta)
    u = ray / torch.linalg.norm(ray, dim=-1, keepdim=True)
    off_ray = delta - (delta * u).sum(-1, keepdim=True) * u
    lateral = (off_ray * off_ray).sum(-1) / widths[1] ** 2
    return radial + torch.where(has_virt, lateral, torch.zeros_like(lateral))


def fan_energies(a, b, l0, widths):
    """Harmonic on each metal-virtual and virtual-virtual separation."""
    return ((torch.linalg.norm(a - b, dim=-1) - l0) / widths[2]) ** 2


def metal_coordination_pose_scores(
    coords,
    rot_coord_offset,
    _pose_ind_for_atom,
    first_rot_for_block,
    _first_rot_block_type,
    _block_ind_for_rot,
    _pose_ind_for_rot,
    _block_type_ind_for_rot,
    _n_rots_for_pose,
    _rot_offset_for_pose,
    _n_rots_for_block,
    _rot_offset_for_block,
    _max_n_rots_per_pose,
    site_rows,
    site_d0,
    fan_rows,
    fan_l0,
    widths,
    output_block_pair_energies: bool,
):
    n_poses, max_n_blocks = first_rot_for_block.shape
    block_pair = torch.zeros(
        (n_poses, max_n_blocks, max_n_blocks), dtype=coords.dtype, device=coords.device
    )

    def atom(pose, block, at):
        rot = first_rot_for_block[pose, block].to(torch.int64)
        return coords[rot_coord_offset[rot].to(torch.int64) + at]

    if site_rows.shape[0] > 0:
        pose, mblock, matom, vatom, dblock, datom = site_rows.unbind(1)
        has_virt = vatom >= 0
        metal = atom(pose, mblock, matom)
        donor = atom(pose, dblock, datom)
        virt = atom(pose, mblock, vatom.clamp_min(0))
        e = site_energies(metal, donor, virt, has_virt, site_d0, widths)
        block_pair = block_pair.index_put((pose, mblock, dblock), e, accumulate=True)

    if fan_rows.shape[0] > 0:
        pose, block, a, b = fan_rows.unbind(1)
        e = fan_energies(atom(pose, block, a), atom(pose, block, b), fan_l0, widths)
        block_pair = block_pair.index_put((pose, block, block), e, accumulate=True)

    if site_rows.shape[0] == 0 and fan_rows.shape[0] == 0:
        block_pair = _coordinate_independent_score(coords, block_pair)

    if output_block_pair_energies:
        return block_pair.unsqueeze(0), None
    return block_pair.sum(dim=(1, 2)).unsqueeze(0), None
