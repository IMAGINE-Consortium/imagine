"""
Tools for working with sparse matrices
"""
# %% IMPORTS
# Package imports
import numpy as np
import scipy.sparse as spr

# All declaration
__all__ = ['slogdet']

# %% FUNCTION DEFINITIONS
def minimumSwaps(arr):
    """
    Minimum number of swaps needed to order a
    permutation array
    """
    # Based on
    # https://www.thepoorcoder.com/hackerrank-minimum-swaps-2-solution/
    a = dict(enumerate(arr))
    b = {v:k for k,v in a.items()}
    count = 0
    for i in a:
        x = a[i]
        if x!=i:
            y = b[i]
            a[y] = x
            b[x] = y
            count+=1

    return count


def slogdet(arr):
    """
    Computes the sign and log of the determinant
    of a sparse matrix

    Parameters
    ----------
    arr : array_like
        Sparse matrix (or numpy array)

    Returns
    -------
    sign : int
        Sign of the determinant
    logdet : float
        Natural logarithm of the determinant
    """
    # Based on the discussion in https://stackoverflow.com/a/60982033/4862845
    lu = spr.linalg.splu(arr)

    diagL = lu.L.diagonal()
    diagU = lu.U.diagonal()

    logdet = np.log(np.abs(diagL)).sum() + np.log(np.abs(diagU)).sum()

    sign = np.sign(diagL).prod()*np.sign(diagU).prod()
    sign *= (-1)**(minimumSwaps(lu.perm_r))

    return sign, logdet
