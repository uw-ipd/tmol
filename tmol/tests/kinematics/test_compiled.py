import torch
import attr

from tmol.kinematics.compiled import compiled
from tmol.kinematics.scan_ordering import KinTreeScanData, KinTreeScanOrdering


def test_deriv_passing(torch_device):
    f1s = torch.tensor(
        [
            [1.000, 0.000, 0.000],
            [1.000, 0.000, 0.000],
            [1.000, 0.000, 0.000],
            [1.000, 0.000, 0.000],
            [1.000, 0.000, 0.000],
            [1.000, 0.000, 0.000],
        ]
    ).to(dtype=torch.double, device=torch_device)

    f2s = torch.tensor(
        [
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000],
        ]
    ).to(dtype=torch.double, device=torch_device)
    f1f2s = torch.cat((f1s, f2s), 1)

    # scan paths
    scansList = [
        torch.tensor([0], dtype=torch.int32, device=torch_device),
        torch.tensor([0, 2], dtype=torch.int32, device=torch_device),
    ]
    nodesList = [
        torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=torch_device),
        torch.tensor([2, 4, 2, 5], dtype=torch.int32, device=torch_device),
    ]
    forward_scan_paths = KinTreeScanData(nodes=nodesList, scans=scansList)

    # reverse forward scan paths --> deriv scans
    nodesListR = [
        torch.tensor([5, 2, 4, 2], dtype=torch.int32, device=torch_device),
        torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=torch_device),
    ]
    scansListR = [
        torch.tensor([0, 2], dtype=torch.int32, device=torch_device),
        torch.tensor([0], dtype=torch.int32, device=torch_device),
    ]
    backward_scan_paths = KinTreeScanData(nodes=nodesListR, scans=scansListR)

    ordering = KinTreeScanOrdering(
        forward_scan_paths=forward_scan_paths, backward_scan_paths=backward_scan_paths
    )

    # expected results
    f1f2s_expected = f1f2s.clone()
    for i in range(len(nodesListR)):
        for j in range(scansListR[i].shape[0]):
            scanstart = scansListR[i][j]
            scanstop = nodesListR[i].shape[0]
            if j != scansListR[i].shape[0] - 1:
                scanstop = scansListR[i][j + 1]
            for k in range(scanstart, scanstop - 1):
                parent = nodesListR[i][k].to(torch.long)
                child = nodesListR[i][k + 1].to(torch.long)
                f1f2s_expected[child] = f1f2s_expected[child] + f1f2s_expected[parent]

    compiled.segscan_f1f2s(f1f2s, **attr.asdict(ordering.backward_scan_paths))
    torch.testing.assert_allclose(f1f2s[:, 0], f1f2s_expected[:, 0])
