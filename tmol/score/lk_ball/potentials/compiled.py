from tmol.utility.cpp_extension import load, modulename, relpaths

_compiled = load(modulename(__name__), relpaths(__file__, "compiled.cc"))


def lk_ball_V(*args, **kwargs):
    return _compiled.lk_ball_V[(args[0].device.type, args[0].dtype)](*args, **kwargs)


def lk_ball_dV(*args, **kwargs):
    return _compiled.lk_ball_dV[(args[0].device.type, args[0].dtype)](*args, **kwargs)


def attached_waters_forward(*args, **kwargs):
    return _compiled.attached_waters_forward[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )


def attached_waters_backward(*args, **kwargs):
    return _compiled.attached_waters_backward[(args[0].device.type, args[0].dtype)](
        *args, **kwargs
    )
