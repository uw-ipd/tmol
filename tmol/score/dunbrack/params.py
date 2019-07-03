import attr
import cattr

import numpy
import pandas
import torch

import toolz.functoolz

from tmol.types.array import NDArray
from tmol.types.torch import Tensor
from tmol.types.tensor import TensorGroup
from tmol.types.attrs import ValidateAttrs, ConvertAttrs
from tmol.types.functional import validate_args

from tmol.numeric.bspline import BSplineInterpolation

from tmol.database.scoring.dunbrack_libraries import DunbrackRotamerLibrary

# dunbrack database
@attr.s(auto_attribs=True, slots=True, frozen=True)
class PackedDunbrackDatabase(ConvertAttrs):
    rotameric_prob_tables: Tensor(torch.float)[:, :, :]
    rotameric_bbsteps: Tensor(torch.float)[:, 2]
    rotameric_bbstarts: Tensor(torch.float)[:, 2]

    semirotameric_prob_tables: Tensor(torch.float)[:, :, :, :]
    semirotameric_bbsteps: Tensor(torch.float)[:, 3]
    semirotameric_bbstarts: Tensor(torch.float)[:, 3]

    # all indexed by 'aa_indices'
    # (that is, on AA 'type', at least, dun's definition of such)
    rotind2rotprobind: Tensor(torch.int32)[:, 3 ** 4]
    rotind2rotmeanind: Tensor(torch.int32)[:, 3 ** 4]
    rotind2semirotprobind: Tensor(torch.int32)[:, 3 ** 4]
    nrotchi_aa: Tensor(torch.int32)[:]
    semirotchi_aa: Tensor(torch.int32)[:]


# dunbrack parameters for a particular protein
@attr.s(auto_attribs=True)
class DunbrackParams(TensorGroup):
    bb_indices: Tensor(torch.int32)[:, 2, 4]
    chi_indices: Tensor(torch.int32)[:, 4, 4]
    aa_indices: Tensor(torch.int32)[:]


@attr.s(frozen=True, slots=True, auto_attribs=True)
class DunbrackParamResolver(ValidateAttrs):
    _from_dun_db_cache = {}

    packed_db: PackedDunbrackDatabase

    # aa -> table mapping
    dun_lookup: pandas.DataFrame

    device: torch.device

    @classmethod
    @validate_args
    @toolz.functoolz.memoize(
        cache=_from_dun_db_cache,
        key=lambda args, kwargs: (args[1], args[2].type, args[2].index),
    )
    def from_database(cls, dun_database: DunbrackRotamerLibrary, device: torch.device):
        # setup name to index mapping
        dun_lookup = pandas.DataFrame.from_records(
            cattr.unstructure(dun_database.dun_lookup)
        ).set_index(["residue_name"])
        aaindices = pandas.Index(
            [
                *[f.table_name for f in dun_database.rotameric_libraries],
                *[f.table_name for f in dun_database.semi_rotameric_libraries],
            ]
        )

        dun_lookup["aaidx"] = aaindices.get_indexer(dun_lookup.dun_table_name)

        # compute table sizes
        naaindices = len(aaindices)
        nrottables = 0
        nsemirottables = 0
        for f in dun_database.rotameric_libraries:
            nchi = f.rotameric_data.nchi()
            nrot = f.rotameric_data.nrotamers()
            nrottables += nrot * (2 + 2 * nchi)  # prob/logprob/nchi*(mean/stdev)
        for f in dun_database.semi_rotameric_libraries:
            nrotchi = f.rotameric_data.nchi()
            nrot = f.rotameric_data.nrotamers()  # store all (for now)
            nrottables += 2 * nrot * nrotchi  # nchi*(mean/stdev)
            nsemirottables += 2 * nrot * nrotchi  # prob/logprob

        # note: we assume all rot/semirot tables are the same size
        rottablesize = dun_database.rotameric_libraries[
            0
        ].rotameric_data.rotamer_probabilities.shape[1:]
        semirottablesize = dun_database.semi_rotameric_libraries[
            0
        ].nonrotameric_chi_probabilities.shape[1:]

        # allocate tables (on CPU to start)
        rotameric_prob_tables = torch.empty(
            [nrottables, *rottablesize], dtype=torch.float
        )
        rotameric_bbsteps = torch.empty([nrottables, 2], dtype=torch.float)
        rotameric_bbstarts = torch.empty([nrottables, 2], dtype=torch.float)
        semirotameric_prob_tables = torch.empty(
            [nsemirottables, *semirottablesize], dtype=torch.float
        )
        semirotameric_bbsteps = torch.empty([nsemirottables, 3], dtype=torch.float)
        semirotameric_bbstarts = torch.empty([nsemirottables, 3], dtype=torch.float)
        rotind2rotprobind = torch.full([naaindices, 81], -1, dtype=torch.int32)
        rotind2rotmeanind = torch.full([naaindices, 81], -1, dtype=torch.int32)
        rotind2semirotprobind = torch.full([naaindices, 81], -1, dtype=torch.int32)
        nrotchi_aa = torch.full([naaindices], -1, dtype=torch.int32)
        semirotchi_aa = torch.full([naaindices], -1, dtype=torch.int32)

        # populate tables and derived indices
        cls.fill_rotameric_tables(
            aaindices,
            dun_database.rotameric_libraries,
            dun_database.semi_rotameric_libraries,
            rotameric_prob_tables,
            rotameric_bbsteps,
            rotameric_bbstarts,
            rotind2rotprobind,
            rotind2rotmeanind,
            nrotchi_aa,
        )

        cls.fill_semirotameric_tables(
            aaindices,
            dun_database.semi_rotameric_libraries,
            semirotameric_prob_tables,
            semirotameric_bbsteps,
            semirotameric_bbstarts,
            rotind2semirotprobind,
            semirotchi_aa,
        )

        cls.compute_table_coeffs(rotameric_prob_tables, semirotameric_prob_tables)

        packed_db = PackedDunbrackDatabase(
            rotameric_prob_tables=rotameric_prob_tables.to(device=device),
            rotameric_bbsteps=rotameric_bbsteps.to(device=device),
            rotameric_bbstarts=rotameric_bbstarts.to(device=device),
            semirotameric_prob_tables=semirotameric_prob_tables.to(device=device),
            semirotameric_bbsteps=semirotameric_bbsteps.to(device=device),
            semirotameric_bbstarts=semirotameric_bbstarts.to(device=device),
            rotind2rotprobind=rotind2rotprobind.to(device=device),
            rotind2rotmeanind=rotind2rotmeanind.to(device=device),
            rotind2semirotprobind=rotind2semirotprobind.to(device=device),
            nrotchi_aa=nrotchi_aa.to(device=device),
            semirotchi_aa=semirotchi_aa.to(device=device),
        )

        return cls(packed_db=packed_db, dun_lookup=dun_lookup, device=device)

    # map a rotamer index (0-81) to a table index or -1 if no defined table
    @classmethod
    def rotidx2tblidx(cls, rotlib):
        rotamers = rotlib.rotameric_data.rotamers
        exponents = [x for x in range(rotamers.shape[1])]
        exponents.reverse()
        exponents = torch.tensor(exponents, dtype=torch.int32)
        prods = torch.pow(3, exponents)
        rotinds = torch.sum((rotamers - 1) * prods, 1)
        ri2ti = -1 * torch.ones([3 ** rotamers.shape[1]], dtype=torch.int32)
        ri2ti[rotinds] = torch.arange(rotamers.shape[0], dtype=torch.int32)

        # Andrew, what is this logic for?
        if (
            len(rotlib.rotameric_data.rotamer_alias.shape) == 2
            and rotlib.rotameric_data.rotamer_alias.shape[0] > 0
        ):
            orig_rotids = (
                rotlib.rotameric_data.rotamer_alias[:, 0 : rotamers.shape[1]]
                .clone()
                .detach()
            )
            alt_rotids = (
                rotlib.rotameric_data.rotamer_alias[:, rotamers.shape[1] :]
                .clone()
                .detach()
            )
            orig_inds = torch.sum((orig_rotids - 1) * prods, 1)
            alt_inds = torch.sum((alt_rotids - 1) * prods, 1)
            ri2ti[orig_inds.type(torch.long)] = ri2ti[alt_inds.type(torch.long)]

        return ri2ti

    # calc. spline coeffs.
    @classmethod
    def compute_table_coeffs(cls, rotameric_prob_tables, semirotameric_prob_tables):
        for i in range(rotameric_prob_tables.shape[0]):
            rotameric_prob_tables[i] = BSplineInterpolation.from_coordinates(
                rotameric_prob_tables[i]
            ).coeffs

        for i in range(semirotameric_prob_tables.shape[0]):
            semirotameric_prob_tables[i] = BSplineInterpolation.from_coordinates(
                semirotameric_prob_tables[i]
            ).coeffs

    # fill in tables/indices describing rotameric AAs
    #  (and rotameric chis of semirot AAs)
    @classmethod
    def fill_rotameric_tables(
        cls,
        aaindices,
        rotlibs,
        semirotlibs,
        rotameric_prob_tables,
        rotameric_bbsteps,
        rotameric_bbstarts,
        rotind2rotprobind,
        rotind2rotmeanind,
        nrotchi_aa,
    ):
        rotstartidx = 0

        # rotameric tables
        for rotlib in rotlibs:
            nchi = rotlib.rotameric_data.nchi()
            nrot = rotlib.rotameric_data.nrotamers()
            aaidx = aaindices.get_loc(rotlib.table_name)
            nrotchi_aa[aaidx] = nchi
            rotidx2tblidx = cls.rotidx2tblidx(rotlib)

            ntables = nrot * (2 + 2 * nchi)  # prob/logprob/nchi*(mean/stdev)

            # table parameters
            rotameric_bbstarts[rotstartidx : rotstartidx + ntables] = (
                rotlib.rotameric_data.backbone_dihedral_start * numpy.pi / 180
            )
            rotameric_bbsteps[rotstartidx : rotstartidx + ntables] = (
                rotlib.rotameric_data.backbone_dihedral_step * numpy.pi / 180
            )

            # tables
            # start by interleaving prob and logprob for each rotamer
            probs_corr = rotlib.rotameric_data.rotamer_probabilities.clone().detach()
            probs_corr[probs_corr < 1e-6] = 1e-6

            rotameric_prob_tables[rotstartidx : rotstartidx + 2 * nrot : 2] = probs_corr
            rotameric_prob_tables[
                rotstartidx + 1 : rotstartidx + 2 * nrot : 2
            ] = -1 * torch.log(probs_corr)

            rotind2rotprobind[aaidx, : rotidx2tblidx.shape[0]] = (
                rotstartidx + 2 * rotidx2tblidx
            )  # prob/logprob

            # increment table pointer
            rotstartidx += 2 * nrot

            # next interleave mean and stdev for each rotamer chi
            # (correct bins crossing 180 to always be positive)
            mean_corr = rotlib.rotameric_data.rotamer_means.clone().detach()
            mean_corr[mean_corr < -120] = mean_corr[mean_corr < -120] + 360
            mean_corr *= numpy.pi / 180

            # rotamer tables seperate out chi# and rot#.
            #  Interleave these (rot and then chi)
            nx = mean_corr.shape[1]
            ny = mean_corr.shape[2]
            rotameric_prob_tables[
                rotstartidx : rotstartidx + 2 * nrot * nchi : 2
            ] = mean_corr.permute([0, 3, 1, 2]).reshape(-1, nx, ny)

            stdv_corr = rotlib.rotameric_data.rotamer_stdvs.clone().detach()
            stdv_corr *= numpy.pi / 180
            rotameric_prob_tables[
                rotstartidx + 1 : rotstartidx + 2 * nrot * nchi : 2
            ] = stdv_corr.permute([0, 3, 1, 2]).reshape(-1, nx, ny)

            rotind2rotmeanind[aaidx, : rotidx2tblidx.shape[0]] = (
                rotstartidx + 2 * nchi * rotidx2tblidx
            )  # nchi*(mean/stdev)

            # increment table pointer
            rotstartidx += 2 * nrot * nchi

        # rotameric part of semirotameric tables (means/stdevs for rot. chis)
        for rotlib in semirotlibs:
            nchi = rotlib.rotameric_data.nchi()
            nrot = rotlib.rotameric_data.nrotamers()
            nchirot = rotlib.rotameric_chi_rotamers.shape[0]
            aaidx = aaindices.get_loc(rotlib.table_name)

            nrotchi_aa[aaidx] = nchi - 1

            # table parameters
            rotameric_bbstarts[rotstartidx : rotstartidx + 2 * nrot * nchi] = (
                rotlib.rotameric_data.backbone_dihedral_start * numpy.pi / 180
            )
            rotameric_bbsteps[rotstartidx : rotstartidx + 2 * nrot * nchi] = (
                rotlib.rotameric_data.backbone_dihedral_step * numpy.pi / 180
            )

            # This assumes that:
            #  a) for rotameric data, all combinations of
            #     rotameric-chi rotamers + binned-non-rotameric-chi
            #     are defined
            #  b) the rotamers stored in the rotameric_data are in rot-order
            #
            # note (fd): we are storing full rotamer tables here even though
            #   we only use part of this data in scoring
            rotidx2tblidx = nrot / nchirot * torch.arange(nchirot, dtype=torch.int32)

            # interleave mean and stdev
            # (correct bins crossing 180 to always be positive)
            mean_corr = rotlib.rotameric_data.rotamer_means.clone().detach()
            mean_corr[mean_corr < -120] = mean_corr[mean_corr < -120] + 360
            mean_corr *= numpy.pi / 180

            # rotamer tables seperate out chi# and rot#.
            #  Interleave these (rot and then chi)
            nx = mean_corr.shape[1]
            ny = mean_corr.shape[2]
            rotameric_prob_tables[
                rotstartidx : rotstartidx + 2 * nrot * nchi : 2
            ] = mean_corr.permute([0, 3, 1, 2]).reshape(-1, nx, ny)

            stdv_corr = rotlib.rotameric_data.rotamer_stdvs.clone().detach()
            stdv_corr *= numpy.pi / 180
            rotameric_prob_tables[
                rotstartidx + 1 : rotstartidx + 2 * nrot * nchi : 2
            ] = stdv_corr.permute([0, 3, 1, 2]).reshape(-1, nx, ny)

            rotind2rotmeanind[aaidx, : rotidx2tblidx.shape[0]] = (
                rotstartidx + 2 * nchi * rotidx2tblidx
            )  # nchi*(mean/stdev)

            # increment table pointer
            rotstartidx += 2 * nrot * nchi

    @classmethod
    def fill_semirotameric_tables(
        cls,
        aaindices,
        semirotlibs,
        semirotameric_prob_tables,
        semirotameric_bbsteps,
        semirotameric_bbstarts,
        rotind2semirotprobind,
        semirotchi_aa,
    ):
        rotstartidx = 0
        for rotlib in semirotlibs:
            nchi = rotlib.rotameric_data.nchi()
            nchirot = rotlib.rotameric_chi_rotamers.shape[0]
            aaidx = aaindices.get_loc(rotlib.table_name)
            semirotchi_aa[aaidx] = nchi

            # table parameters
            semirotameric_bbstarts[rotstartidx : rotstartidx + 2 * nchirot, :2] = (
                rotlib.rotameric_data.backbone_dihedral_start * numpy.pi / 180
            )
            semirotameric_bbstarts[rotstartidx : rotstartidx + 2 * nchirot, 2] = (
                rotlib.non_rot_chi_start * numpy.pi / 180
            )
            semirotameric_bbsteps[rotstartidx : rotstartidx + 2 * nchirot, :2] = (
                rotlib.rotameric_data.backbone_dihedral_step * numpy.pi / 180
            )
            semirotameric_bbsteps[rotstartidx : rotstartidx + 2 * nchirot, 2] = (
                rotlib.non_rot_chi_step * numpy.pi / 180
            )

            # tables
            # interleave prob and logprob for each rotamer
            semirotameric_prob_tables[
                rotstartidx : rotstartidx + 2 * nchirot : 2
            ] = rotlib.nonrotameric_chi_probabilities
            semirotameric_prob_tables[
                rotstartidx + 1 : rotstartidx + 2 * nchirot : 2
            ] = -1 * torch.log(rotlib.nonrotameric_chi_probabilities)

            rotind2semirotprobind[aaidx, :nchirot] = rotstartidx + 2 * torch.arange(
                nchirot
            )

            # increment table pointer
            rotstartidx += 2 * nchirot

    def resolve_dunbrack_parameters(
        self,
        res_names: NDArray(object)[:],
        phis: Tensor(torch.int32)[:, 5],
        psis: Tensor(torch.int32)[:, 5],
        chis: Tensor(torch.int32)[:, 6],
        torch_device: torch.device,
    ) -> DunbrackParams:
        # 1) find all residues where phi/psi are undefied and set all indices to -1
        phimask, _ = (phis[:, 1:] == -1).max(dim=1)
        phis[phimask, 1:] = -1
        psimask, _ = (psis[:, 1:] == -1).max(dim=1)
        psis[psimask, 1:] = -1

        # 2) fill DunbrackParams tensors
        maxres = res_names.shape[0]
        bbidx = torch.full([maxres, 2, 4], -1, dtype=torch.int32, device=torch_device)
        chiidx = torch.full([maxres, 4, 4], -1, dtype=torch.int32, device=torch_device)
        bbidx[phis[:, 0].to(torch.long), 0, :] = phis[:, 1:]
        bbidx[psis[:, 0].to(torch.long), 1, :] = psis[:, 1:]
        chiidx[chis[:, 0].to(torch.long), chis[:, 1].to(torch.long), :] = chis[:, 2:]

        aaidx = self.dun_lookup.index.get_indexer(res_names)
        taaidx = torch.tensor(
            self.dun_lookup.iloc[aaidx[aaidx >= 0], :]["aaidx"], dtype=torch.int32
        )
        dun_defined = torch.from_numpy(aaidx) >= 0

        # 3) select subset where potentials are defined
        tbbidx = bbidx[dun_defined, :, :]
        tchiidx = chiidx[dun_defined, :, :]

        return DunbrackParams(
            bb_indices=tbbidx.to(device=torch_device),
            chi_indices=tchiidx.to(device=torch_device),
            aa_indices=taaidx.to(device=torch_device),
        )
