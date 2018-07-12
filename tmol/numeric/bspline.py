import torch
import math

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from typing import Optional

# This file contains code for N-dimensional b-spline interpolation
#
# B-splines can be used for interpolation, using only as much space
# as the original data, after the original data is processed to produce
# coefficients for the polynomials. When interpolating, b-splines read from
# as much memory as bicublic-spline interpolation does (e.g. 16 values
# in 2D, 64 values in 3D), but they read from a wider number of grid cells
# to do so, instead of making each grid cell contain more entries. For this
# reason, the memory footprint for B-splines is substantially lower than
# that for bicuplic spline interpolation. (Catmull-Rom splines are similarly
# low-memory overhead, but not as good).


@validate_args
def init_causal_coeff(
        coeffs: Tensor(torch.float)[:, :], pole: Tensor(torch.float)
):
    """inplace calculation of [:,0] coefficients.
    currently, initialization corresponds to periodic boundaries
    (if one were to add alternate boundary conditions, this would be the place)
    assumes inputs are an M x N tensor, and splines are computed in the 2nd dim"""

    N = coeffs.shape[1]
    tol = 1e-7
    horiz = math.ceil(math.log(tol) / math.log(abs(pole.item())))

    zn = pole.clone()
    if (horiz < N):
        for i in range(1, horiz):
            coeffs[:, 0] += zn * coeffs[:, N - i]
            zn *= pole
    else:
        for i in range(1, N):
            coeffs[:, 0] += zn * coeffs[:, N - i]
            zn *= pole
        coeffs[:, 0] = (coeffs[:, 0] / (1 - zn))


@validate_args
def init_anticausal_coeff(
        coeffs: Tensor(torch.float)[:, :], pole: Tensor(torch.float)
):
    """inplace calculation of [:,N] coefficients.
    currently, initialization corresponds to periodic boundaries
    (if one were to add alternate boundary conditions, this would be the place)
    assumes inputs are an M x N tensor, and splines are computed in the 2nd dim
    this initialization corresponds to periodic boundaries"""
    N = coeffs.shape[1]
    tol = 1e-7
    horiz = math.ceil(math.log(tol) / math.log(abs(pole.item())))

    zn = pole.clone()
    if (horiz < N):
        for i in range(horiz):
            coeffs[:, N - 1] += zn * coeffs[:, i]
            zn *= pole
        coeffs[:, N - 1] = -pole * coeffs[:, N - 1]
    else:
        for i in range(N):
            coeffs[:, N - 1] += zn * coeffs[:, i]
            zn *= pole
        coeffs[:, N - 1] = -pole * coeffs[:, N - 1] / (1 - zn)


@validate_args
def convert_interp_coeffs(
        coeffs: Tensor(torch.float)[:, :], poles: Tensor(torch.float)[:]
):
    """interpolation coefficients in one dimension
    assumes input an M x N tensor, and interpolation is carried out in the 2nd dim
    returns resulting coefficients
    """
    N = coeffs.shape[1]
    retval = coeffs.clone()
    if (N == 1):
        return retval

    lmbda = torch.prod((1 - poles) * (1 - 1 / poles))
    retval = lmbda * retval

    for pole in poles:
        # init ( [:,0] ) coeffs, in-place
        init_causal_coeff(retval, pole)
        # forward sweep
        for i in range(1, N):
            retval[:, i] += pole * retval[:, i - 1]

        # final ( [:,N-1] ) coeffs, in-place
        init_anticausal_coeff(retval, pole)
        # backward sweep
        for i in range(N - 2, -1, -1):
            retval[:, i] = pole * (retval[:, i + 1] - retval[:, i])

    return retval


@validate_args
def compute_coeffs(coords: Tensor(torch.float),
                   degree: int) -> Tensor(torch.float):
    """Full N-D interpolation coefficients"""

    coeffs = coords.clone()

    if degree == 2:
        poles = torch.tensor([math.sqrt(8.0) - 3.0])
    elif degree == 3:
        poles = torch.tensor([math.sqrt(3.0) - 2.0])
    elif degree == 4:
        poles = torch.tensor([
            math.sqrt(664.0 - math.sqrt(438976.0)) + math.sqrt(304.0) - 19.0,
            math.sqrt(664.0 + math.sqrt(438976.0)) - math.sqrt(304.0) - 19.0
        ])
    elif degree == 5:
        poles = torch.tensor([
            math.sqrt(135.0 / 2.0 - math.sqrt(17745.0 / 4.0)) +
            math.sqrt(105.0 / 4.0) - 13.0 / 2.0,
            math.sqrt(135.0 / 2.0 + math.sqrt(17745.0 / 4.0)) -
            math.sqrt(105.0 / 4.0) - 13.0 / 2.0
        ])
    else:
        die('Invalid spline degree')

    ndims = len(coords.shape)
    for dim in range(ndims):
        # permutation to make 'dim' the last dimension and flatten all other dimensions
        coeffs = coeffs.transpose(dim, ndims - 1)
        coeffs = coeffs.reshape(-1, coords.shape[dim])

        # compute the interp coeffs along last dimension
        coeffs = convert_interp_coeffs(coeffs, poles)

        # back to the original shape
        coeffs = coeffs.reshape(coords.shape).transpose(dim, ndims - 1)

    return coeffs


@validate_args
def interpolate(
        coeffs: Tensor(torch.float),
        degree: int,
        X: Tensor(torch.float)[:, :],
        Y: Optional[Tensor(torch.long)[:, :]] = None
) -> Tensor(torch.float)[:]:
    """b-spline interpolation function
    takes precalculated coefficients as input, and returns value at given grid index

    If Y is provided, it is treated as providing indexes for (leading) non-interpolating dimensions;
    e.g. if the Ramachandran map is 20x36x36, then the Y tensor could state which of the
    20 amino acids were being read from and then the X tensor would provided the (shifted+scaled)
    phi and psi values.

    Y must have the same number of rows as X (their first dimensions must be the same size)
    """

    ndims = len(coeffs.shape)
    n_non_interp_dims = 0
    if Y is not None:
        n_non_interp_dims = Y.shape[1]
        ndims -= n_non_interp_dims
    nx = X.shape[0]

    # we need to compute over a ((degree+1)^N) box
    #  - we first calculate indices seperately in a (degree+1) x N array
    #  - then we expand to the full ((degree+1)^N) box and dot-prod

    # calculate interpolation indices
    baseline = torch.floor(X - (degree - 1) / 2.0)
    offset = X - baseline
    indx_bydim = torch.arange(degree + 1).reshape(1, -1, 1) + baseline.reshape(
        -1, 1, ndims
    )

    # construct weight matrix
    wts_bydim = torch.empty((nx, degree + 1, ndims),
                            dtype=coeffs.dtype,
                            device=coeffs.device)
    if degree == 2:
        w = X - indx_bydim[:, 1, :]
        wts_bydim[:, 1, :] = 3.0 / 4.0 - w * w
        wts_bydim[:, 2, :] = (1.0 / 2.0) * (w - wts_bydim[:, 1, :] + 1.0)
        wts_bydim[:, 0, :] = 1.0 - wts_bydim[:, 1, :] - wts_bydim[:, 2, :]
    elif degree == 3:
        w = X - indx_bydim[:, 1, :]
        wts_bydim[:, 3, :] = (1.0 / 6.0) * w * w * w
        wts_bydim[:, 0, :] = ((1.0 / 6.0) +
            (1.0 / 2.0) * w * (w - 1.0) - wts_bydim[:, 3, :]) # yapf: disable
        wts_bydim[:, 2, :] = w + wts_bydim[:, 0, :] - 2.0 * wts_bydim[:, 3, :]
        wts_bydim[:, 1, :] = ( 1.0 - wts_bydim[:, 0, :] -
            wts_bydim[:, 2, :] - wts_bydim[:, 3, :] ) # yapf: disable
    elif degree == 4:
        w = X - indx_bydim[:, 2, :]
        w2 = w * w
        t = (1.0 / 6.0) * w2
        wts_bydim[:, 0, :] = 1.0 / 2.0 - w
        wts_bydim[:, 0, :] *= wts_bydim[:, 0, :]
        wts_bydim[:, 0, :] *= (1.0 / 24.0) * wts_bydim[:, 0, :]
        t0 = w * (t - 11.0 / 24.0)
        t1 = 19.0 / 96.0 + w2 * (1.0 / 4.0 - t)
        wts_bydim[:, 1, :] = t1 + t0
        wts_bydim[:, 3, :] = t1 - t0
        wts_bydim[:, 4, :] = wts_bydim[:, 0, :] + t0 + (1.0 / 2.0) * w
        wts_bydim[:, 2, :] = (1.0 - wts_bydim[:, 0, :] -
            wts_bydim[:, 1, :] - wts_bydim[:, 3, :] - wts_bydim[:, 4, :])# yapf: disable
    elif degree == 5:
        w = X - indx_bydim[:, 2, :]
        w2 = w * w
        wts_bydim[:, 5, :] = (1.0 / 120.0) * w * w2 * w2
        w2 -= w
        w4 = w2 * w2
        w -= 1.0 / 2.0
        t = w2 * (w2 - 3.0)
        wts_bydim[:, 0, :] = ((1.0 / 24.0) * (1.0 / 5.0 + w2 + w4) -
            wts_bydim[:, 5, :]) # yapf: disable
        t0 = (1.0 / 24.0) * (w2 * (w2 - 5.0) + 46.0 / 5.0)
        t1 = (-1.0 / 12.0) * w * (t + 4.0)
        wts_bydim[:, 2, :] = t0 + t1
        wts_bydim[:, 3, :] = t0 - t1
        t0 = (1.0 / 16.0) * (9.0 / 5.0 - t)
        t1 = (1.0 / 24.0) * w * (w4 - w2 - 5.0)
        wts_bydim[:, 1, :] = t0 + t1
        wts_bydim[:, 4, :] = t0 - t1
    else:
        die('Invalid spline degree')

    # apply periodicity
    # this is only valid for periodic boundaries
    indx_bydim = torch.fmod(
        indx_bydim.long(),
        torch.tensor(coeffs.shape[n_non_interp_dims:], dtype=torch.long)
    )

    # now expand to ndim-dimensional box
    # there might be a better way to do this
    # indx_expand = coeffs
    wts_expand = torch.full((nx, 1), 1.0, dtype=torch.float)

    inds = torch.zeros((nx, 1), dtype=torch.long)

    interp_dims_offset = 1
    for dim in range(ndims):
        #indx_expand = torch.index_select( indx_expand, dim, indx_bydim[:,dim] )
        inds = coeffs.shape[dim + n_non_interp_dims] * inds.reshape(
            nx, -1, 1
        ) + indx_bydim[:, :, dim].reshape(nx, 1, -1)
        interp_dims_offset *= coeffs.shape[dim + n_non_interp_dims]

        wts_expand = wts_expand.reshape(
            nx, -1, 1
        ) * wts_bydim[:, :, dim].reshape(nx, 1, -1)
        newdims = (degree + 1) * torch.ones(dim + 1)
    if Y is not None:
        non_interp_dims_offset = interp_dims_offset
        # create a tuple of -1 followed by 1s of the right length for broadcasting
        # against the inds tensor
        newshape = (-1, ) + (1, ) * (len(inds.shape) - 1)
        for ii in range(Y.shape[1] - 1, -1, -1):
            # now increment the indices
            inds += (non_interp_dims_offset * Y[:, ii]).reshape(newshape)
            non_interp_dims_offset *= Y.shape[ii]

    # ... and do the dot product
    retval = torch.sum(
        wts_expand.reshape(nx, -1) *
        coeffs.view(-1)[inds].reshape(nx, -1).squeeze(), 1
    )

    return retval
