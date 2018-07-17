"""N-dimensional B-spline interpolation with periodic boundary conditions.

Reference:
    Thévenaz, Philippe, Thierry Blu, and Michael Unser.
    "Interpolation revisited [medical images application]."
    IEEE Transactions on medical imaging 19.7 (2000): 739-758.

    http://bigwww.epfl.ch/publications/thevenaz0002.pdf

B-splines can be used for interpolation, using only as much space
as the original data, after the original data is processed to produce
coefficients for the polynomials. When interpolating, B-splines read from
as much memory as bicublic-spline interpolation does (e.g. 16 values
in 2D, 64 values in 3D), but they read from a wider number of grid cells
to do so, instead of making each grid cell contain more entries. For this
reason, the memory footprint for B-splines is substantially lower than
that for bicuplic spline interpolation. (Catmull-Rom splines are similarly
low-memory overhead, but do not fit the data as cleanly).

To use these B-splines, construct a BSplineInterpolation object using the
`from_coordinates` function, passing in the tensor of coordinates that
should be interpolated, indicating the degree of the spline (3 for
the equivalent of bicubic spline interpolation), and indicating the
number of dimensions in the coordinate tensor that are indexing rather
than interpolating. (Indexing dimensions must appear first). Indexing
dimensions allow stacks of coordinate tensors to be interpolated
simultaneously as might be useful, e.g., if one had 20 different 36x36
tables as one does when computing the Ramachandran potential.

Interpolation is performed where the input X values must be in the range of (0, |X_i|]
for dimension i -- if, e.g., you are interpolating dihedrals in degrees with a 10 degree
step size in the range (-180, 180], then add 180 to the dihedral shifting to the range
[0, 360) and then divide by 10 to produce an interpolation value in the range (0, 36].
Another way to say this is that the code assumes unit distance between interpolation
points.
"""

import torch
import math
import attr

from tmol.types.torch import Tensor
from tmol.types.functional import validate_args
from typing import Optional


@attr.s(auto_attribs=True, frozen=True, slots=True)
class BSplineDegree:
    @classmethod
    @validate_args
    def empty_wts_bydim(
            cls, ndims: int, coeffs: Tensor(torch.float),
            X: Tensor(torch.float)[:, :]
    ) -> Tensor(torch.float)[:, :, :]:
        """Allocate wts_bydim tensor with the dtype and device following
        coeffs's example
        """
        return torch.empty((X.shape[0], cls.degree + 1, ndims),
                           dtype=coeffs.dtype,
                           device=coeffs.device)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class BSplineDegree2(BSplineDegree):
    degree = 2
    poles = torch.tensor([math.sqrt(8.0) - 3.0])

    @classmethod
    @validate_args
    def compute_wts_bydim(
            cls, ndims: int, coeffs: Tensor(torch.float),
            X: Tensor(torch.float)[:, :],
            indx_bydim: Tensor(torch.float)[:, :, :]
    ) -> Tensor(torch.float)[:, :, :]:

        wts_bydim = cls.empty_wts_bydim(ndims, coeffs, X)

        w = X - indx_bydim[:, 1, :]
        wts_bydim[:, 1, :] = 3.0 / 4.0 - w * w
        wts_bydim[:, 2, :] = (1.0 / 2.0) * (w - wts_bydim[:, 1, :] + 1.0)
        wts_bydim[:, 0, :] = 1.0 - wts_bydim[:, 1, :] - wts_bydim[:, 2, :]
        return wts_bydim


@attr.s(auto_attribs=True, frozen=True, slots=True)
class BSplineDegree3(BSplineDegree):
    degree = 3
    poles = torch.tensor([math.sqrt(3.0) - 2.0])

    @classmethod
    @validate_args
    def compute_wts_bydim(
            cls, ndims: int, coeffs: Tensor(torch.float),
            X: Tensor(torch.float)[:, :],
            indx_bydim: Tensor(torch.float)[:, :, :]
    ) -> Tensor(torch.float)[:, :, :]:

        wts_bydim = cls.empty_wts_bydim(ndims, coeffs, X)

        w = X - indx_bydim[:, 1, :]
        wts_bydim[:, 3, :] = (1.0 / 6.0) * w * w * w
        wts_bydim[:, 0, :] = ((1.0 / 6.0) +
            (1.0 / 2.0) * w * (w - 1.0) - wts_bydim[:, 3, :]) # yapf: disable
        wts_bydim[:, 2, :] = w + wts_bydim[:, 0, :] - 2.0 * wts_bydim[:, 3, :]
        wts_bydim[:, 1, :] = (1.0 - wts_bydim[:, 0, :] -
            wts_bydim[:, 2, :] - wts_bydim[:, 3, :]) # yapf: disable

        return wts_bydim


@attr.s(auto_attribs=True, frozen=True, slots=True)
class BSplineDegree4(BSplineDegree):
    degree = 4
    poles = torch.tensor([
        math.sqrt(664.0 - math.sqrt(438976.0)) + math.sqrt(304.0) - 19.0,
        math.sqrt(664.0 + math.sqrt(438976.0)) - math.sqrt(304.0) - 19.0
    ])

    @classmethod
    @validate_args
    def compute_wts_bydim(
            cls, ndims: int, coeffs: Tensor(torch.float),
            X: Tensor(torch.float)[:, :],
            indx_bydim: Tensor(torch.float)[:, :, :]
    ) -> Tensor(torch.float)[:, :, :]:

        wts_bydim = cls.empty_wts_bydim(ndims, coeffs, X)

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

        return wts_bydim


@attr.s(auto_attribs=True, frozen=True, slots=True)
class BSplineDegree5(BSplineDegree):
    degree = 5
    poles = torch.tensor([
        math.sqrt(135.0 / 2.0 - math.sqrt(17745.0 / 4.0)) +
        math.sqrt(105.0 / 4.0) - 13.0 / 2.0,
        math.sqrt(135.0 / 2.0 + math.sqrt(17745.0 / 4.0)) -
        math.sqrt(105.0 / 4.0) - 13.0 / 2.0
    ])

    @classmethod
    @validate_args
    def compute_wts_bydim(
            cls, ndims: int, coeffs: Tensor(torch.float),
            X: Tensor(torch.float)[:, :],
            indx_bydim: Tensor(torch.float)[:, :, :]
    ) -> Tensor(torch.float)[:, :, :]:

        wts_bydim = cls.empty_wts_bydim(ndims, coeffs, X)

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

        return wts_bydim


# Mapping from (integer) degree to the set of BSplineDegrees that
# are supported
bsplines_by_degree = {
    b.degree: b
    for b in [
        BSplineDegree2,
        BSplineDegree3,
        BSplineDegree4,
        BSplineDegree5,
    ]
}


@attr.s(auto_attribs=True, frozen=True, slots=True)
class BSplineInterpolation:
    """Class for performing bspline interpolation with periodic boundary conditions.

    Construct an instance of this class using the `from_coordinates` function, handing
    it a (possibly stacked) table of coordinates that should be interpolated by the
    splines, the degree of the spline that should be constructed, and (optionally)
    the number of dimensions in the coordinates tensor that are "indexing dimensions"
    and not interpolation dimensions. The indexing dimensions should appear as the
    most significant dimensions and the interpolating dimensions should appear as the
    least significant dimensions.

    Once constructed, the `interpolate` method can be given a tensor of coordinates
    `X` (and if the original coordinate tensor had indexing dimensions, a tensor of
    indices `Y`) to produce a tensor of interpolated values.
    """

    degree: int
    coeffs: Tensor(torch.float)
    n_interp_dims: int
    n_index_dims: int

    @classmethod
    @validate_args
    def from_coordinates(
            cls, coords: Tensor(torch.float), degree: int,
            n_index_dims: int = 0
    ):
        """Construct a BSplineInterpolation instance from  the input coordinates
        (i.e. the data to be interpolated)

        This code handles splines of arbitrary dimension.

        The input coordinate tensor holds the values that should be interpolated.
        The first n_index_dims are not interpolated, but are rather used to hold
        indexing dimensions so that stacks of tables can be interpolated together.
        (E.g, as the 20x2 set of 36x36 tables are organized for the Ramachandran
        term.)
        """

        coeffs = coords.clone()
        original_shape = coeffs.shape

        n_interp_dims = len(coords.shape) - n_index_dims

        interp_shape = coeffs.shape[n_index_dims:]
        coeffs = coeffs.reshape(-1, *interp_shape)

        if degree not in bsplines_by_degree:
            raise ValueError(
                "Invalid b-spline degree of %d requested; available degrees are %s"
                % (
                    degree,
                    ", ".join([str(k) for k in bsplines_by_degree.keys()])
                )
            )

        bspdeg = bsplines_by_degree[degree]

        for i in range(coeffs.shape[0]):
            # slice "row" i
            icoeffs = coeffs.narrow(0, i, 1).squeeze(dim=0)
            icoeffs_out = icoeffs
            for dim in range(n_interp_dims):
                # permutation to make 'dim' the last dimension
                icoeffs = icoeffs.transpose(dim, n_interp_dims - 1)
                trans_shape = icoeffs.shape
                # now flatten all other dimensions
                icoeffs = icoeffs.reshape(-1, interp_shape[dim])

                # compute the interp coeffs along last dimension
                icoeffs = cls._convert_interp_coeffs(icoeffs, bspdeg.poles)

                # restore to the translated shape and then transpose
                # the last dimension to where it belongs
                icoeffs = icoeffs.reshape(trans_shape).transpose(
                    dim, n_interp_dims - 1
                )
                icoeffs_out[:] = icoeffs

        coeffs = coeffs.reshape(original_shape)
        return cls(
            degree=degree,
            coeffs=coeffs,
            n_interp_dims=n_interp_dims,
            n_index_dims=n_index_dims
        )

    @validate_args
    def interpolate(
            self,
            X: Tensor(torch.float)[:, :],
            Y: Optional[Tensor(torch.long)[:, :]] = None
    ) -> Tensor(torch.float)[:]:
        """B-spline interpolation function

        Takes precalculated coefficients as input, and returns value at given grid index.
        Input X values must be in the range [0..|X_i|) for each dimension i.

        If Y is provided, it is treated as providing indexes for (leading) non-interpolating dimensions;
        e.g. if the Ramachandran map is 20x36x36, then the Y tensor could state which of the
        20 amino acids were being read from and then the X tensor would provided the (shifted+scaled)
        phi and psi values.

        Y must have the same number of rows as X (their first dimensions must be the same size)
        """

        assert len(X.shape) == 2
        assert X.shape[1] == self.n_interp_dims
        assert Y is None or len(Y.shape) == 2
        assert Y is None or X.shape[0] == Y.shape[0]
        assert ((Y is None and self.n_index_dims == 0) or
                (Y is not None and Y.shape[1] == self.n_index_dims)) # yapf: disable

        bspdeg = bsplines_by_degree[self.degree]
        nx = X.shape[0]

        # we need to compute over a ((bspdeg.degree+1)^N) box
        #  - we first calculate indices seperately in a (bspdeg.degree+1) x N array
        #  - then we expand to the full ((bspdeg.degree+1)^N) box and dot-prod

        # calculate interpolation indices
        baseline = torch.floor(X - (bspdeg.degree - 1) / 2.0)
        indx_bydim = torch.arange(
            bspdeg.degree + 1, device=self.coeffs.device
        ).reshape(1, -1, 1) + baseline.reshape(-1, 1, self.n_interp_dims)

        # construct weight matrix -- this varies depending on the degree of the
        # bspline, and therefore is delegated to the BSplineDegree class
        wts_bydim = bspdeg.compute_wts_bydim(
            self.n_interp_dims, self.coeffs, X, indx_bydim
        )

        # apply periodicity.
        # this is only valid for periodic boundaries.
        # `remainder` and not `fmod` so that all results are non-negative.
        indx_bydim = torch.remainder(
            indx_bydim.long(),
            torch.tensor(
                self.coeffs.shape[self.n_index_dims:],
                dtype=torch.long,
                device=self.coeffs.device
            )
        )

        # now expand to (n_interp_dims)-dimensional box.
        # there might be a better way to do this
        wts_expand = torch.full((nx, 1), 1.0, dtype=torch.float)

        inds = torch.zeros((nx, 1), dtype=torch.long)

        interp_dims_offset = 1
        for dim in range(self.n_interp_dims):
            inds = self.coeffs.shape[dim + self.n_index_dims] * inds.reshape(
                nx, -1, 1
            ) + indx_bydim[:, :, dim].reshape(nx, 1, -1)
            interp_dims_offset *= self.coeffs.shape[dim + self.n_index_dims]

            wts_expand = wts_expand.reshape(
                nx, -1, 1
            ) * wts_bydim[:, :, dim].reshape(nx, 1, -1)

        if Y is not None:
            non_interp_dims_offset = interp_dims_offset
            # create a tuple of -1 followed by 1s of the right length for broadcasting
            # against the inds tensor
            newshape = (-1, ) + (1, ) * (len(inds.shape) - 1)
            for ii in range(self.n_index_dims - 1, -1, -1):
                # now increment the indices
                inds += (non_interp_dims_offset * Y[:, ii]).reshape(newshape)
                non_interp_dims_offset *= self.coeffs.shape[ii]

        # ... and do the dot product
        retval = torch.sum(
            wts_expand.reshape(nx, -1) *
            self.coeffs.view(-1)[inds].reshape(nx, -1), 1
        )

        return retval

    @staticmethod
    @validate_args
    def _init_causal_coeff(
            coeffs: Tensor(torch.float)[:, :], pole: Tensor(torch.float)
    ):
        """Helper function for b-spline coefficient calculation

        inplace calculation of [:,0] coefficients.
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

    @staticmethod
    @validate_args
    def _init_anticausal_coeff(
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

    @classmethod
    @validate_args
    def _convert_interp_coeffs(
            cls, coeffs: Tensor(torch.float)[:, :],
            poles: Tensor(torch.float)[:]
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
            cls._init_causal_coeff(retval, pole)
            # forward sweep
            for i in range(1, N):
                retval[:, i] += pole * retval[:, i - 1]

            # final ( [:,N-1] ) coeffs, in-place
            cls._init_anticausal_coeff(retval, pole)
            # backward sweep
            for i in range(N - 2, -1, -1):
                retval[:, i] = pole * (retval[:, i + 1] - retval[:, i])

        return retval
