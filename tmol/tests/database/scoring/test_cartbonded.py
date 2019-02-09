from collections import Counter


def test_cartbonded_defs(default_database):
    db = default_database.scoring.cartbonded

    bondlength_counts = Counter((n.res, n.atm1, n.atm2) for n in db.length_parameters)

    for res, atm1, atm2 in bondlength_counts:
        assert (
            bondlength_counts[(res, atm1, atm2)] == 1
        ), f"Duplicate cartbonded type parameter: {(res,atm1,atm2)}"
        assert (
            atm1 == atm2 or bondlength_counts[(res, atm2, atm1)] == 0
        ), f"Reversed cartbonded type parameter: {(res,atm1,atm2)}"

    bondangle_counts = Counter(
        (n.res, n.atm1, n.atm2, n.atm3) for n in db.angle_parameters
    )

    for res, atm1, atm2, atm3 in bondangle_counts:
        assert (
            bondangle_counts[(res, atm1, atm2, atm3)] == 1
        ), f"Duplicate cartbonded type parameter: {(res,atm1,atm2,atm3)}"
        assert (
            bondangle_counts[(res, atm3, atm2, atm1)] == 0
        ), f"Reversed cartbonded type parameter: {(res,atm1,atm2,atm3)}"

    bondtorsion_counts = Counter(
        (n.res, n.atm1, n.atm2, n.atm3, n.atm4) for n in db.torsion_parameters
    )

    for res, atm1, atm2, atm3, atm4 in bondtorsion_counts:
        assert (
            bondtorsion_counts[(res, atm1, atm2, atm3, atm4)] == 1
        ), f"Duplicate cartbonded type parameter: {(res,atm1,atm2,atm3,atm4)}"
        assert (
            bondtorsion_counts[(res, atm4, atm3, atm2, atm1)] == 0
        ), f"Reversed cartbonded type parameter: {(res,atm1,atm2,atm3,atm4)}"

    bondimproper_counts = Counter(
        (n.res, n.atm1, n.atm2, n.atm3, n.atm4) for n in db.improper_parameters
    )

    for res, atm1, atm2, atm3, atm4 in bondimproper_counts:
        assert (
            bondimproper_counts[(res, atm1, atm2, atm3, atm4)] == 1
        ), f"Duplicate cartbonded type parameter: {(res,atm1,atm2,atm3,atm4)}"
        assert (
            bondimproper_counts[(res, atm4, atm3, atm2, atm1)] == 0
        ), f"Reversed cartbonded type parameter: {(res,atm1,atm2,atm3,atm4)}"

    hxltorsion_parameters = Counter(
        (n.res, n.atm1, n.atm2, n.atm3, n.atm4) for n in db.hxltorsion_parameters
    )

    for res, atm1, atm2, atm3, atm4 in hxltorsion_parameters:
        assert (
            hxltorsion_parameters[(res, atm1, atm2, atm3, atm4)] == 1
        ), f"Duplicate cartbonded type parameter: {(res,atm1,atm2,atm3,atm4)}"
        assert (
            hxltorsion_parameters[(res, atm4, atm3, atm2, atm1)] == 0
        ), f"Reversed cartbonded type parameter: {(res,atm1,atm2,atm3,atm4)}"
