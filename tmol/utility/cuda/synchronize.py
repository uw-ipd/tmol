import torch


def synchronize_if_cuda_available():
    """Calls to torch.cuda.synchronize() that are not preceeded by
    a call to torch.cuda.is_available lead to program exits on
    machines that do not have CUDA. Instead of duplicating this
    if statement everywhere -- here's a simple function.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
