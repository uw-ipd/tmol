import torch


def convert_float64(args):
    for i in range(len(args)):
        if isinstance(args[i], torch.Tensor) and args[i].dtype == torch.float32:
            args[i] = args[i].double()
