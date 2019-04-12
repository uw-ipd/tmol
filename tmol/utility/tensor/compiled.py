from tmol.utility.cpp_extension import load, relpaths, modulename, cuda_if_available
from typing import List
from tmol.types.functional import validate_args

_compiled = load(
    modulename(__name__),
    cuda_if_available(
        relpaths(
            __file__, ["compiled.pybind.cpp"]  # "compiled.cpu.cpp", "compiled.cuda.cu"]
        )
    ),
)


@validate_args
def create_tensor_collection(tensor_list: List):
    if len(tensor_list) == 0:
        raise ValueError(
            "tensor list passed to create_tensor_collection must be non-empty"
        )

    for tensor in tensor_list:
        assert len(tensor.shape) == len(
            tensor_list[0].shape
        ), "All tensors in tensor_list must be the same dimension"

    if len(tensor_list[0].shape) == 1:
        return _create_tensor_collection1(tensor_list)
    elif len(tensor_list[0].shape) == 2:
        return _create_tensor_collection2(tensor_list)
    elif len(tensor_list[0].shape) == 3:
        return _create_tensor_collection3(tensor_list)
    elif len(tensor_list[0].shape) == 4:
        return _create_tensor_collection4(tensor_list)
    else:
        raise ValueError(
            "highest supported dimension for TensorCollection is"
            + " 4; input tensors are of dimension "
            + str(len(tensor_list[0].shape))
        )


def _create_tensor_collection1(tensor_list: List):
    return _compiled.create_tensor_collection1[
        (tensor_list[0][0].device.type, tensor_list[0][0].dtype)
    ](tensor_list)


def _create_tensor_collection2(tensor_list: List):
    return _compiled.create_tensor_collection2[
        (tensor_list[0][0].device.type, tensor_list[0][0].dtype)
    ](tensor_list)


def _create_tensor_collection3(tensor_list: List):
    return _compiled.create_tensor_collection3[
        (tensor_list[0][0].device.type, tensor_list[0][0].dtype)
    ](tensor_list)


def _create_tensor_collection4(tensor_list: List):
    return _compiled.create_tensor_collection4[
        (tensor_list[0][0].device.type, tensor_list[0][0].dtype)
    ](tensor_list)
