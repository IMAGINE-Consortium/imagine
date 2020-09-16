# %% IMPORTS
# Built-in imports
from math import floor, log10

# Package imports
import astropy.units as apu

# All declaration
__all__ = ['adjust_error_intervals', 'is_notebook', 'unit_checker']


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


def adjust_error_intervals(value, errlo, errup, sdigits=2, return_ndec=False):
    r"""
    Takes the value of a quantity `value` with associated errors `errlo` and
    `errup`; and prepares them to be reported as
    :math:`v^{+err\,up}_{-err\,down}`. This is done by adjusting the number
    of decimal places of all the argumetns so that the errors have at least
    `sdigits` significant digits. Optionally, this number of decimal places
    may be returned.


    Parameters
    ----------
    value : int or float or astropy.Quantity
        Value of quantity.
    errlo, errup : int or float or astropy.Quantity
        Associated lower and upper errors of `value`.
    sdigits : int, optional
        Minimum number of significant digits in the errors
    return_ndec : bool, optional
        If True, also returns the number of decimal points used

    Returns
    -------
    value : float
        Rounded value
    errlo, errup : float
        Assimetric error values
    n : int
        If `return_ndec` is `True`, the number of decimal places is returned
    """
    unit, [value, errlo, errup] = unit_checker(None, [value, errlo, errup])
    get_rounding = lambda x: -int(floor(log10(abs(x)))) + (sdigits - 1)

    if unit is None:
        unit = 1.0

    n = max(get_rounding(errlo), get_rounding(errup))

    value, errlo, errup = (round(x,n)*unit for x in (value, errlo, errup))

    if not return_ndec:
        return value, errlo, errup
    else:
        return value, errlo, errup, n


def unit_checker(unit, list_of_quant):
    """
    Checks the consistency of units of a list of quantities, converting them
    all to the same units, if needed.

    Parameters
    ----------
    unit : astropy.Unit
        Unit to be used for the quantities in the list. If set to `None`, the
        units of the first list item are used.
    list_of_quant : list
        List of quantities to be checked.

    Returns
    -------
    unit : astropy.Unit
        The common unit used
    list_of_values :
        Contains the quantities of `list_of_quant` converted to floats using
        the common unit `unit`
    """
    ul = []
    for uq in list_of_quant:
        if isinstance(uq, apu.Quantity):
            if unit is None:
                unit = uq.unit
            else:
                uq.to(unit)
            ul.append(uq.to_value(unit))
        else:
            ul.append(uq)
    return unit, ul
