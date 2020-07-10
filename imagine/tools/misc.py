# %% IMPORTS
# Built-in imports
from math import floor, log10

# All declaration
__all__ = ['adjust_error_intervals', 'is_notebook']


# %% FUNCTION DEFINITIONS
def is_notebook():
    """
    Finds out whether python is running in a Jupyter notebook
    or as a shell.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def adjust_error_intervals(value, errlo, errup, sdigits=2):
    r"""
    Takes the value of a quantity `value` with associated errors `errlo` and
    `errup`; and prepares them to be reported as
    :math:`v^{+err\,up}_{-err\,down}`.

    Parameters
    ----------
    value : int or float
        Value of quantity.
    errlo, errup : int or float
        Associated lower and upper errors of `value`.

    Returns
    -------
    value : float
        Rounded value
    errlo, errup : float
        Assimetric error values

    """

    get_rounding = lambda x: -int(floor(log10(abs(x)))) + (sdigits - 1)

    n = max(get_rounding(errlo), get_rounding(errup))

    value, errlo, errup = (round(x,n) for x in (value, errlo, errup))

    return(value, errlo, errup)
