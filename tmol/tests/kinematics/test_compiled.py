import torch
import attr

from tmol.kinematics.compiled import compiled
from tmol.kinematics.scan_ordering import KinTreeScanData


def test_deriv_passing(torch_device):
    f1f2s = torch.ones((6, 6), dtype=torch.double, device=torch_device)

    # deriv scans
    nodesList = torch.tensor(
        [5, 2, 4, 2, 3, 2, 1, 0], dtype=torch.int32, device=torch_device
    )
    scansList = torch.tensor([0, 2, 0], dtype=torch.int32, device=torch_device)
    gensList = torch.tensor(
        [[0, 0], [4, 2]], dtype=torch.int32, device=torch.device("cpu")
    )
    backward_scan_paths = KinTreeScanData(
        nodes=nodesList, scans=scansList, gens=gensList
    )

    f1f2s_in = f1f2s.clone()
    compiled.segscan_f1f2s(f1f2s, **attr.asdict(backward_scan_paths))

    check_scan_results(f1f2s, f1f2s_in, nodesList, scansList, gensList)


def test_deriv_passing_1k_atms(torch_device):
    N = 1000
    f1f2s = torch.ones((N, 6), dtype=torch.double, device=torch_device)

    # deriv scans
    nodesList = torch.arange(N - 1, -1, -1, dtype=torch.int32, device=torch_device)
    scansList = torch.tensor([0], dtype=torch.int32, device=torch_device)
    gensList = torch.tensor([[0, 0]], dtype=torch.int32, device=torch.device("cpu"))
    backward_scan_paths = KinTreeScanData(
        nodes=nodesList, scans=scansList, gens=gensList
    )

    f1f2s_in = f1f2s.clone()
    compiled.segscan_f1f2s(f1f2s, **attr.asdict(backward_scan_paths))

    check_scan_results(f1f2s, f1f2s_in, nodesList, scansList, gensList)


def check_scan_results(f1f2s, f1f2s_in, nodesList, scansList, gensList):
    f1f2s_expected = f1f2s_in.clone()

    gensList = gensList.to(device=nodesList.device)

    for i in range(len(gensList) - 1):
        scanstart = gensList[i][1]
        scanstop = gensList[i + 1][1]
        for j in range(scanstart, scanstop):
            nodestart = gensList[i][0] + scansList[j]
            nodestop = gensList[i + 1][0]
            if j < scanstop - 1:
                nodestop = gensList[i][0] + scansList[j + 1]
            for k in range(nodestart, nodestop - 1):
                parent = nodesList[k].to(torch.long)
                child = nodesList[k + 1].to(torch.long)
                f1f2s_expected[child] = f1f2s_expected[child] + f1f2s_expected[parent]

    torch.testing.assert_allclose(f1f2s, f1f2s_expected)
