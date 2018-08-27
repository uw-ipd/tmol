import numba


@numba.jit(nopython=True)
def iterative_refold(hts, parent):
    for ii in range(1, hts.shape[0]):
        hts[ii, :, :] = hts[parent[ii], :, :] @ hts[ii, :, :]


@numba.jit(nopython=True)
def iterative_f1f2_summation(f1f2s, parent):
    for ii in range(f1f2s.shape[0] - 1, 0, -1):
        f1f2s[parent[ii], :] += f1f2s[ii, :]
