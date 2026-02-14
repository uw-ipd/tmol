from tmol.kinematics.compiled._compiled_inverse_kin import inverse_kin as _inverse_kin_dispatch


def inverse_kin(*args, **kwargs):
    return _inverse_kin_dispatch[(args[0].device.type, args[0].dtype)](*args, **kwargs)
