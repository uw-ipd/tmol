import pytest

from tmol.kinematics.builder import KinematicBuilder


@pytest.fixture
def ubq_kintree(ubq_system):
    return KinematicBuilder().append_connected_component(
        *KinematicBuilder.bonds_to_connected_component(0, ubq_system.bonds)
    ).kintree
