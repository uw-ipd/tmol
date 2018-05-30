import pytest

import torch
import torch.cuda

cuda_available = torch.cuda.is_available()

requires_cuda = pytest.mark.skipif(not cuda_available, reason="Requires cuda.")
