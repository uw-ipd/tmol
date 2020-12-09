import pytest

import tmol.system.restypes


@pytest.fixture
def default_restype_set():
    return tmol.system.restypes.ResidueTypeSet.get_default()


@pytest.fixture
def fresh_default_restype_set(default_database):
    """Fresh ResidueTypeSet constructed for each test
    """
    return tmol.system.restypes.ResidueTypeSet.from_database(default_database.chemical)


@pytest.fixture()
def rts_ubq_res(fresh_default_restype_set, ubq_res):
    import attr

    rts = fresh_default_restype_set

    return [
        attr.evolve(
            res,
            residue_type=next(
                rt for rt in rts.residue_types if rt.name == res.residue_type.name
            ),
        )
        for res in ubq_res
    ]
