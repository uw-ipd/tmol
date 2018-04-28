import pytest

from tmol.viewer import SystemViewer


def test_system_viewer_smoke(ubq_system):
    SystemViewer(ubq_system)
    SystemViewer(ubq_system, style="stick", mode="cdjson")
    with pytest.raises(NotImplementedError):
        SystemViewer(ubq_system, mode="pdb")


def test_residue_viewer_smoke(ubq_res):
    SystemViewer(ubq_res[0], mode="cdjson")

    with pytest.raises(NotImplementedError):
        SystemViewer(ubq_res[1], mode="pdb")
