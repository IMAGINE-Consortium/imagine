from math import floor, log10

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


def adjust_error_intervals(value, low, up, digits=2):
    r"""
    Takes values of a quantity at -1sigma (low), median (value) and +1sigma (up)
    and prepares them to be reported as :math:`v^{+err\,up}_{-err\,down}`.

    Parameters
    ----------
    value, low, up : float or int
        Value of the number

    Returns
    -------
    value : float
        Rounded value
    errlow, errup : float
        Assimetric error values
    """
    errlow = low-value
    errup = up-value

    get_rounding = lambda x: -int(floor(log10(abs(x)))) + (digits - 1)

    n = max(get_rounding(errlow), get_rounding(errup))

    value, errlow, errup = (round(x,n) for x in (value, errlow, errup))

    return value, errlow, errup


