import numpy
import numba

from tmol.types.array import NDArray
from tmol.io.canonical_ordering import (
    ordered_canonical_aa_types,
    ordered_canonical_aa_atoms,
)
from tmol.utility.auto_number import AutoNumber


# set some constants based on the fixed, canonical ordering of atoms in the canonical AAs
# defined in
his_co_aa_ind = ordered_canonical_aa_types.index("HIS")
his_ND1_in_co = ordered_canonical_aa_atoms["HIS"].index("ND1")
his_NE2_in_co = ordered_canonical_aa_atoms["HIS"].index("NE2")
his_HD1_in_co = ordered_canonical_aa_atoms["HIS"].index("HD1")
his_HE2_in_co = ordered_canonical_aa_atoms["HIS"].index("HE2")
his_HN_in_co = ordered_canonical_aa_atoms["HIS"].index("HN")
his_NH_in_co = ordered_canonical_aa_atoms["HIS"].index("NH")
his_NN_in_co = ordered_canonical_aa_atoms["HIS"].index("NN")
his_CG_in_co = ordered_canonical_aa_atoms["HIS"].index("CG")


class HisTautomerResolution(AutoNumber):
    his_taut_missing_atoms = ()
    his_taut_HD1 = ()
    his_taut_HE2 = ()
    his_taut_NH_is_ND1 = ()
    his_taut_NN_is_ND1 = ()
    his_taut_HD1_HE2 = ()  # future
    his_taut_unresolved = ()  # future


def resolve_his_tautomerization(
    res_types: NDArray[numpy.int32][:, :],
    coords: NDArray[numpy.float32][:, :, :, 3],
    atom_is_present: NDArray[numpy.int32][:, :, :],
) -> NDArray[numpy.int32][:, :]:
    his_pose_ind, his_res_ind = numpy.nonzero(res_types == his_co_aa_ind)
    return resolve_his_tautomerization_numba(
        res_types, his_pose_ind, his_res_ind, coords, atom_is_present
    )


@numba.jit(nopython=True)
def resolve_his_tautomerization_numba(
    res_types: NDArray[numpy.int32][:, :],
    his_pose_ind: NDArray[numpy.int32][:],
    his_res_ind: NDArray[numpy.int32][:],
    coords: NDArray[numpy.float32][:, :, :, 3],
    atom_is_present: NDArray[numpy.float32][:, :, :],
):
    """Resolve which of four cases we have for HIS's tautomerization state:
    a. HIS HD1 is provided (and HE2 is not) and we are in tautomerization state HD1,
       HisTautomerResolution.his_taut_HD1
    b. HIS HE2 is provided (and HD1 is not) and we are in tautomerization state HE2
    c. HIS HN, ND1 and NE2 are provided (and neither HD1 nor HE2 is) and we select the
       tautomerization state based on which nitrogen HN is closest to
    d. HIS HN, NH, and NN are provided and we select the tautomerization state based
       on the distances between both NH and NN to CG; the one closer to CG is taken
        as ND1.
    """

    his_taut = numpy.zeros_like(res_types)
    for i in range(his_pose_ind.shape[0]):
        ip = his_pose_ind[i]
        ir = his_res_ind[i]
        ND1_present = atom_is_present[ip, ir, his_ND1_in_co]
        NE2_present = atom_is_present[ip, ir, his_NE2_in_co]
        HD1_present = atom_is_present[ip, ir, his_HD1_in_co]
        HE2_present = atom_is_present[ip, ir, his_HE2_in_co]
        HN_present = atom_is_present[ip, ir, his_HN_in_co]
        NH_present = atom_is_present[ip, ir, his_NH_in_co]
        NN_present = atom_is_present[ip, ir, his_NN_in_co]
        CG_present = atom_is_present[ip, ir, his_CG_in_co]

        state = HisTautomerResolution.his_taut_unresolved.value

        if HD1_present and not HE2_present:
            state = HisTautomerResolution.his_taut_HD1.value
        elif HE2_present and not HD1_present:
            state = HisTautomerResolution.his_taut_HE2.value
        elif (
            HN_present
            and not HD1_present
            and not HE2_present
            and ND1_present
            and NE2_present
        ):
            dis2_ND1 = numpy.sum(
                numpy.square(
                    coords[ip, ir, his_ND1_in_co] - coords[ip, ir, his_HN_in_co]
                )
            )
            dis2_NE2 = numpy.sum(
                numpy.square(
                    coords[ip, ir, his_NE2_in_co] - coords[ip, ir, his_HN_in_co]
                )
            )
            if dis2_ND1 < dis2_NE2:
                state = HisTautomerResolution.his_taut_HD1.value
            else:
                state = HisTautomerResolution.his_taut_HE2.value
        elif NH_present and NN_present and HN_present and CG_present:
            dis2_NH = numpy.sum(
                numpy.square(
                    coords[ip, ir, his_NH_in_co] - coords[ip, ir, his_CG_in_co]
                )
            )
            dis2_NN = numpy.sum(
                numpy.square(
                    coords[ip, ir, his_NN_in_co] - coords[ip, ir, his_CG_in_co]
                )
            )
            if dis2_NH < dis2_NN:
                state = HisTautomerResolution.his_taut_NH_is_ND1.value
            else:
                state = HisTautomerResolution.his_taut_NN_is_ND1.value
        his_taut[ip, ir] = int(state)
    return his_taut
