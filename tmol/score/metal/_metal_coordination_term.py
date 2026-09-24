"""Metal coordination restraints."""

from itertools import combinations

import numpy
import torch

from tmol.chemical import RefinedResidueType
from tmol.database import ParameterDatabase
from tmol.database.chemical import ideal_distances, metal_table
from tmol.pose import PackedBlockTypes, PoseStack

from .._annotation_cache import AnnotationKey, cached_annotation, store_annotation
from .._energy_term import EnergyTerm

# kernel sentinels for metal_conn_virt
NOT_A_SITE = -2
NO_VIRTUAL = -1


class MetalCoordinationEnergyTerm(EnergyTerm):
    """Hold each donor on its site's vertex ray and each site fan rigid.

    Per occupied site: a harmonic on the metal-donor distance and on the
    donor's displacement off the ray from the metal through the site virtual,
    plus a constant well depth. Untemplated ions keep only the distance. The
    fan term holds metal and virtual separations at ideal; it is zero whenever
    the fan is built from its icoors.

    A site is occupied when its connection on the metal is filled; the donor
    is the atom on the partner's side of that connection.
    """

    def __init__(self, param_db: ParameterDatabase, device: torch.device):
        super().__init__(param_db=param_db, device=device)
        self.device = device
        self.params = param_db.scoring.metal_coordination
        table = metal_table()
        self.ion_for_atom_type = {ion["atom_type"]: ion for ion in table["ions"]}
        self.donor_radii = table["donor_radii"]
        self.donor_keys = tuple(self.donor_radii)
        self.vertices_for = {g["name"]: g["vertices"] for g in table["geometries"]}
        self._packed_annotation_key = AnnotationKey.from_sources(
            self.params, settings=(self.device,)
        )

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

    def setup_packed_block_types(self, packed_block_types: PackedBlockTypes):
        super().setup_packed_block_types(packed_block_types)
        cached = cached_annotation(
            packed_block_types,
            "_metal_coordination_annotation",
            self._packed_annotation_key,
        )
        if cached is not None:
            return cached

        pbt = packed_block_types
        atom_types = {at.name: at for at in pbt.chem_db.atom_types}
        n_keys = len(self.donor_keys)
        metal_atom = numpy.full(pbt.n_types, -1, dtype=numpy.int32)
        conn_virt = numpy.full((pbt.n_types, pbt.max_n_conn), NOT_A_SITE, numpy.int32)
        conn_key = numpy.full((pbt.n_types, pbt.max_n_conn), -1, dtype=numpy.int32)
        site_params = numpy.zeros((pbt.n_types, n_keys, 4), dtype=numpy.float32)
        fans = [[] for _ in range(pbt.n_types)]

        for i, bt in enumerate(pbt.active_block_types):
            for c, conn in enumerate(bt.connections):
                atom = bt.atoms[bt.atom_to_idx[conn.atom]]
                conn_key[i, c] = self.donor_key(atom_types[atom.atom_type])
            if not bt.metal_sites or bt.metal_sites[0].internal_satisfiers:
                continue
            site = bt.metal_sites[0]
            metal = bt.atom_to_idx[site.metal_atom]
            metal_type = bt.atoms[metal].atom_type
            metal_atom[i] = metal
            virts = [bt.atom_to_idx[v] for v in site.site_virts]
            for k, name in enumerate(site.site_connections):
                conn_virt[i, bt.connection_to_cidx[name]] = (
                    virts[k] if virts else NO_VIRTUAL
                )
            dist = ideal_distances(self.ion_for_atom_type[metal_type], self.donor_radii)
            radial_sd, lateral_sd = self.params.widths(metal_type)
            for k, key in enumerate(self.donor_keys):
                depth = self.params.well_depth(metal_type, key)
                site_params[i, k] = (dist[key], depth, radial_sd, lateral_sd)
            if virts:
                ideal = self.ideal_fan(bt, site)
                for a, b in combinations(list(ideal), 2):
                    fans[i].append((a, b, numpy.linalg.norm(ideal[a] - ideal[b])))

        max_fan = max(1, max(len(f) for f in fans))
        fan_atoms = numpy.full((pbt.n_types, max_fan, 2), -1, dtype=numpy.int32)
        fan_params = numpy.zeros((pbt.n_types, max_fan, 2), dtype=numpy.float32)
        for i, rows in enumerate(fans):
            for j, (a, b, l0) in enumerate(rows):
                fan_atoms[i, j] = (a, b)
                fan_params[i, j] = (l0, self.params.global_parameters.fan_sd)

        def t(x):
            return torch.from_numpy(x).to(self.device)

        fields = {
            "metal_coordination_metal_atom": t(metal_atom),
            "metal_coordination_conn_virt": t(conn_virt),
            "metal_coordination_conn_key": t(conn_key),
            "metal_coordination_site_params": t(site_params),
            "metal_coordination_fan_atoms": t(fan_atoms),
            "metal_coordination_fan_params": t(fan_params),
        }
        for name, value in fields.items():
            setattr(pbt, name, value)
        return store_annotation(
            pbt,
            "_metal_coordination_annotation",
            self._packed_annotation_key,
            tuple(fields.values()),
            fields=tuple(fields),
        )

    def donor_key(self, atom_type):
        """Index of an atom type's metal-donor distance key, or -1."""
        for name in (atom_type.name, atom_type.element):
            if name in self.donor_keys:
                return self.donor_keys.index(name)
        return -1

    def ideal_fan(self, bt, site):
        """Ideal positions of the metal and its site virtuals, by atom index."""
        d = {ic.name: ic.d for ic in bt.icoors}
        verts = numpy.asarray(self.vertices_for[site.geometry], dtype=numpy.float64)
        verts /= numpy.linalg.norm(verts, axis=1, keepdims=True)
        out = {bt.atom_to_idx[site.metal_atom]: numpy.zeros(3)}
        for vertex, name in zip(verts, site.site_virts):
            out[bt.atom_to_idx[name]] = vertex * d[name]
        return out

    def pose_score_term_is_invariant_zero(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types
        bt = pose_stack.block_type_ind64
        is_metal = pbt.metal_coordination_metal_atom[bt.clamp_min(0)] >= 0
        return not bool((is_metal & (bt >= 0)).any())

    def get_pose_score_term_function(self):
        from tmol.score.metal.potentials import metal_coordination_pose_scores

        return metal_coordination_pose_scores

    def get_rotamer_score_term_function(self):
        from tmol.score.metal.potentials import metal_coordination_rotamer_scores

        return metal_coordination_rotamer_scores

    def get_score_term_attributes(self, pose_stack: PoseStack):
        pbt = pose_stack.packed_block_types
        self.check_donors(pose_stack)
        return [
            pose_stack.inter_residue_connections,
            pbt.conn_atom,
            pbt.metal_coordination_metal_atom,
            pbt.metal_coordination_conn_virt,
            pbt.metal_coordination_conn_key,
            pbt.metal_coordination_site_params,
            pbt.metal_coordination_fan_atoms,
            pbt.metal_coordination_fan_params,
        ]

    def check_donors(self, pose_stack: PoseStack):
        """Every filled site's donor atom must have a metal-donor distance."""
        pbt = pose_stack.packed_block_types
        bt = pose_stack.block_type_ind64
        irc = pose_stack.inter_residue_connections64
        is_site = (pbt.metal_coordination_conn_virt[bt.clamp_min(0)] != NOT_A_SITE) & (
            bt >= 0
        ).unsqueeze(-1)
        filled = is_site & (irc[..., 0] >= 0)
        pose, block, conn = torch.nonzero(filled, as_tuple=True)
        if pose.numel() == 0:
            return
        partner, partner_conn = irc[pose, block, conn].unbind(-1)
        key = pbt.metal_coordination_conn_key[bt[pose, partner], partner_conn]
        if bool((key < 0).any()):
            raise ValueError("a metal site is bonded to an atom that is not a donor")
