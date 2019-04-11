import torch
import attr

from tmol.kinematics.compiled import compiled
from tmol.kinematics.scan_ordering import KinTreeScanData


def test_deriv_passing(torch_device):
    f1f2s = torch.ones((6, 6), dtype=torch.double, device=torch_device)

    # deriv scans
    nodesList = [
        torch.tensor([5, 2, 4, 2], dtype=torch.int32, device=torch_device),
        torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=torch_device),
    ]
    scansList = [
        torch.tensor([0, 2], dtype=torch.int32, device=torch_device),
        torch.tensor([0], dtype=torch.int32, device=torch_device),
    ]
    backward_scan_paths = KinTreeScanData(nodes=nodesList, scans=scansList)

    f1f2s_in = f1f2s.clone()
    compiled.segscan_f1f2s(f1f2s, **attr.asdict(backward_scan_paths))

    check_scan_results(f1f2s, f1f2s_in, nodesList, scansList)


def test_deriv_passing_big(torch_device):
    N = 1000
    f1f2s = torch.ones((N, 6), dtype=torch.double, device=torch_device)

    # deriv scans
    nodesList = [torch.arange(N - 1, -1, -1, dtype=torch.int32, device=torch_device)]
    scansList = [torch.tensor([0], dtype=torch.int32, device=torch_device)]
    backward_scan_paths = KinTreeScanData(nodes=nodesList, scans=scansList)

    f1f2s_in = f1f2s.clone()
    compiled.segscan_f1f2s(f1f2s, **attr.asdict(backward_scan_paths))

    check_scan_results(f1f2s, f1f2s_in, nodesList, scansList)


def check_scan_results(f1f2s, f1f2s_in, nodesList, scansList):
    f1f2s_expected = f1f2s_in.clone()
    for i in range(len(nodesList)):
        for j in range(scansList[i].shape[0]):
            scanstart = scansList[i][j]
            scanstop = nodesList[i].shape[0]
            if j != scansList[i].shape[0] - 1:
                scanstop = scansList[i][j + 1]
            for k in range(scanstart, scanstop - 1):
                parent = nodesList[i][k].to(torch.long)
                child = nodesList[i][k + 1].to(torch.long)
                f1f2s_expected[child] = f1f2s_expected[child] + f1f2s_expected[parent]

    print(f1f2s[:30, 0])
    print(f1f2s_expected[:30, 0])

    torch.testing.assert_allclose(f1f2s, f1f2s_expected)
