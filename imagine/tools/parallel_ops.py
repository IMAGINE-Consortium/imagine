"""
Interface module which allows automatically switching between the routines in
the :py:mod:`imagine.tools.mpi_helper` module and their:py:mod:`numpy`
or pure Python equivalents, depending on the contents of
:py:data:`imagine.rc['distributed_arrays']`
"""

# %% IMPORTS
# Package imports
from e13tools import add_to_all
import numpy as np

# IMAGINE imports
from imagine.tools import mpi_helper as m, rc

# All declaration
__all__ = []


# %% FUNCTION DEFINITIONS
@add_to_all
def pshape(data):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_shape` or
    :py:meth:`numpy.ndarray.shape`
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_shape(data)
    else:
        return data.shape


@add_to_all
def prosecutor(data):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_prosecutor` or *nothing*
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        m.mpi_prosecutor(data)


@add_to_all
def pmean(data):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_mean` or :py:func:`numpy.mean`
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_mean(data)
    else:
        return (np.mean(data, axis=0)).reshape(1, -1)


@add_to_all
def ptrans(data):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_mean` or :py:meth:`numpy.ndarray.T`
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_trans(data)
    else:
        return data.T


@add_to_all
def pmult(left, right):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_mult` or :py:meth:`numpy.matmul`
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_mult(left, right)
    else:
        return left @ right


@add_to_all
def ptrace(data):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_trace` or :py:func:`numpy.trace`
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_trace(data)
    else:
        return np.trace(data)

@add_to_all
def pdiag(data):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_diag` or :py:func:`numpy.diagonal`
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_diag(data)
    else:
        return data.diagonal


@add_to_all
def peye(size):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_eye` or :py:func:`numpy.eye`
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_eye(size)
    else:
        return np.eye(size)


@add_to_all
def distribute_matrix(full_matrix):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_distribute_matrix` or *nothing*
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_distribute_matrix(full_matrix)
    else:
        return full_matrix


@add_to_all
def plu_solve(operator, source):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_lu_solve` or :py:func:`numpy.linalg.solve`
    depending on :py:data:`imagine.rc['distributed_arrays']`.

    Notes
    -----
    In the non-distributed case, the source is transposed before the calculation
    """
    if rc['distributed_arrays']:
        return m.mpi_lu_solve(operator, source)
    else:
        return np.linalg.solve(operator, source.T)


@add_to_all
def pslogdet(data):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_slogdet` or :py:func:`numpy.linalg.slogdet`
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_slogdet(data)
    else:
        return np.linalg.slogdet(data)


@add_to_all
def pglobal(data):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_global` or *nothing*
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_global(data)
    else:
        return data


@add_to_all
def plocal(data):
    """
    :py:func:`imagine.tools.mpi_helper.mpi_local` or *nothing*
    depending on :py:data:`imagine.rc['distributed_arrays']`.
    """
    if rc['distributed_arrays']:
        return m.mpi_local(data)
    else:
        return data
